# SnapKV 完整调用链详解


## 第一部分：评估流程启动 (`evaluation/evaluate.py`)

### 0️⃣ **入口函数** - `EvaluationRunner.run_evaluation()`
```python
def run_evaluation(self):
    """评估的主入口函数，协调整个评估流程"""
    output_dir = self._setup_directories()           # ① 创建输出目录
    results_dir = self.config.get_results_dir(...)   # ② 获取结果保存路径
    
    # 检查是否已有结果，避免重复评估
    if predictions_filename.exists() and metrics_filename.exists():
        return
    
    self._setup_press()                              # ③ 初始化 Press（SnapKV）
    self._setup_model_pipeline()                     # ④ 加载模型和 Pipeline
    self._load_and_prepare_dataset()                 # ⑤ 加载数据集
    
    self._run_inference()                            # ⑥ 🔥 核心推理（调用 SnapKV）
    self._save_results(predictions_filename)         # ⑦ 保存预测结果
    self._calculate_and_save_metrics(...)            # ⑧ 计算并保存指标
    self.config.save_config(config_filename)         # ⑨ 保存配置文件
```

**作用**: 总指挥，按顺序调用所有子函数完成评估。

---

### ① `_setup_directories()`
```python
def _setup_directories(self) -> Path:
    output_dir = Path(self.config.output_dir)  # 默认 "./results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```
**作用**: 创建输出目录（如 `./results`），用于保存评估结果。

---

### ② `get_results_dir(output_dir)`
```python
def get_results_dir(self, output_dir: Path) -> Path:
    # 根据配置参数生成唯一的结果子目录
    # 格式：dataset__model__press__compression_ratio
    # 例如：longbench-e__hotpotqa_e__Meta-Llama-3.1-8B-Instruct__snapkv__0.50
    components = [
        self.dataset,                    # "longbench-e"
        self.model.replace("/", "--"),   # "Meta-Llama-3.1-8B-Instruct"
        self.press_name,                 # "snapkv"
        f"{self.compression_ratio:.2f}", # "0.50"
    ]
    dir_name = "__".join(filter(None, components))
    config_dir = output_dir / dir_name
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
```
**作用**: 为本次评估创建唯一的子目录，避免不同配置的结果相互覆盖。

---

### ③ `_setup_press()`
```python
def _setup_press(self):
    press = PRESS_REGISTRY[self.config.press_name]  # 获取 SnapKVPress 实例
    
    # 为 SnapKV 设置压缩率（如 0.5 表示保留 50% 的 Token）
    if hasattr(press, "compression_ratio"):
        press.compression_ratio = self.config.compression_ratio
    
    self.press = press
```
**作用**: 初始化 SnapKV Press，设置压缩率参数（例如从 `debug_config.yaml` 读取的 0.5）。

**关键**: 这里创建的 `press` 对象会在第 ⑥ 步传入 Pipeline。

---

### ④ `_setup_model_pipeline()`
```python
def _setup_model_pipeline(self):
    model_name = self.config.model  # "/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct"
    device = self.config.device or "auto"
    
    # 加载我们自定义的 Pipeline（注册在 kvpress/pipeline.py 中）
    self.pipeline = pipeline(
        "kv-press-text-generation",  # 自定义 Pipeline 名称
        model=model_name,
        device=device,
        model_kwargs={...}
    )
    self.pipeline.model.eval()
```
**作用**: 
- 加载 Llama 模型和 Tokenizer
- 创建 `KVPressTextGenerationPipeline` 实例（我们自定义的 Pipeline）
- 将模型设为评估模式（`eval()`）

**关键**: 这个 Pipeline 会接收 Press 对象，并在推理时激活 KV 压缩。

---

### ⑤ `_load_and_prepare_dataset()`
```python
def _load_and_prepare_dataset(self):
    dataset_name = self.config.dataset  # "longbench-e"
    data_dir = self.config.data_dir     # "hotpotqa_e"
    
    # 从 HuggingFace 加载数据集
    df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()
    
    # 如果设置了 fraction < 1.0，则随机采样（用于快速调试）
    if self.config.fraction < 1.0:
        df = df.sample(frac=self.config.fraction, random_state=self.config.seed)
    
    # 如果启用 query_aware，将问题拼接到 context 后面
    if self.config.query_aware:
        df["context"] = df["context"] + df["question"]
        df["question"] = ""
    
    self.df = df
```
**作用**: 
- 加载评估数据集（如 LongBench-E 的 HotpotQA 子任务）
- 应用采样（`fraction=0.01` 表示只用 1% 数据快速测试）
- 处理查询感知压缩（将问题拼接到上下文）

**数据格式**:
```python
df = pd.DataFrame({
    "context": ["长文本1", "长文本2", ...],      # 输入上下文（如一篇长文档）
    "question": ["问题1", "问题2", ...],        # 需要回答的问题
    "answer": ["答案1", "答案2", ...],          # 正确答案（用于评估）
    "max_new_tokens": [50, 50, ...],           # 每个问题的最大生成长度
    "answer_prefix": ["", "", ...],            # 答案前缀（可选）
})
```

---

### ⑥ `_run_inference()` 🔥 **核心推理函数**
```python
@torch.inference_mode()
def _run_inference(self):
    self.df["predicted_answer"] = None
    
    # 按 context 分组（同一个 context 可能有多个问题）
    df_context_grouped = self.df.groupby("context")
    
    for context, df_group in tqdm(df_context_grouped, desc="Running Inference"):
        questions = df_group["question"].to_list()
        max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
        answer_prefix = df_group["answer_prefix"].iloc[0]
        
        # 🎯 调用 Pipeline（这里会触发 SnapKV 压缩）
        output = self.pipeline(
            context,                                # 长文本输入
            questions=questions,                    # 问题列表
            answer_prefix=answer_prefix,            # 答案前缀
            press=self.press,                       # SnapKVPress 实例
            max_new_tokens=max_new_tokens,          # 最大生成长度
            max_context_length=self.config.max_context_length,
        )
        
        # 保存预测结果
        self.df.loc[df_group.index, "predicted_answer"] = output["answers"]
        self.df.loc[df_group.index, "compression_ratio"] = self.press.compression_ratio
        
        torch.cuda.empty_cache()  # 清理显存
```
**作用**: 
- 遍历数据集中的每个 context
- 调用 `self.pipeline()` 进行推理（**这一步会触发 SnapKV 的 KV 压缩**）
- 保存模型生成的答案

**关键点**:
- **相同 context 的多个问题只需要处理一次 context**（共享压缩后的 KV Cache）
- `self.pipeline()` 内部会调用 `_forward()` 方法（见下一部分）

---

### ⑦ `_save_results(predictions_filename)`
```python
def _save_results(self, save_filename: Path):
    # 保存预测结果到 CSV 文件
    self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
        str(save_filename), index=False
    )
```
**作用**: 将预测结果保存为 `predictions.csv`，包含问题、真实答案、预测答案、压缩率等。

---

### ⑧ `_calculate_and_save_metrics(metrics_filename)`
```python
def _calculate_and_save_metrics(self, save_filename: Path):
    scorer = SCORER_REGISTRY[self.config.dataset]  # 获取对应数据集的评分器
    
    # 计算指标（如准确率、F1 分数等）
    metrics = scorer(self.df)
    
    # 保存到 JSON 文件
    with open(str(save_filename), "w") as f:
        json.dump(metrics, f, indent=4)
```
**作用**: 使用数据集特定的评分器计算性能指标（如 LongBench 的 F1、准确率等），保存为 `metrics.json`。

---

### ⑨ `save_config(config_filename)`
```python
def save_config(self, config_filename: Path):
    # 将评估配置保存为 YAML 文件
    with open(config_filename, "w") as f:
        yaml.dump(asdict(self), f, default_flow_style=False)
```
**作用**: 保存本次评估的完整配置（模型、数据集、Press、压缩率等），确保结果可复现。

---

## 第二部分：Pipeline 层 - KV 压缩的准备 (`kvpress/pipeline.py`)

### 1️⃣ **Pipeline 入口** - `KVPressTextGenerationPipeline.__call__()`
```python
pipeline(context, questions=..., press=self.press, ...)
  │
  ├─→ _sanitize_parameters()    # 参数预处理
  ├─→ preprocess()               # 分词和截断
  ├─→ _forward()                 # 🔥 核心前向传播（这里触发压缩）
  └─→ postprocess()              # 解码生成的 Token
```
**作用**: 
- Pipeline 的统一入口（Transformers 的标准 API）
- 按顺序调用预处理 → 前向传播 → 后处理

---

### 2️⃣ **前向传播核心** - `_forward()`
```python
def _forward(self, input_tensors, max_new_tokens, press, cache):
    """
    分两个阶段：
    1. Prefill：处理完整 context，应用 SnapKV 压缩 KV Cache
    2. Decode：基于压缩后的 Cache 生成答案（逐字生成）
    """
    
    context_ids = input_tensors["context_ids"]  # 分词后的 context
    cache = DynamicCache()  # 创建空的 KV Cache
    
    # ========== Prefill 阶段（压缩 KV Cache） ==========
    with press(self.model):  # 🎯 注册 SnapKV 的 Hook 到所有层
        self.model.model(
            input_ids=context_ids,      # 输入完整 context
            past_key_values=cache,      # 传入空 cache，会被填充
        )
    # 此时 cache 中存储的是压缩后的 KV（每层 500 个 token，而非 1000 个）
    
    # ========== Decode 阶段（生成答案） ==========
    answers = []
    for question_ids in input_tensors["questions_ids"]:
        answer = self.generate_answer(
            question_ids=question_ids,   # 问题的 token IDs
            cache=cache,                 # 使用压缩后的 cache
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        answers.append(answer)
    
    return answers
```

**作用**: 
1. **Prefill 阶段**: 
   - 使用 `with press(self.model)` 激活 SnapKV 的 Hook
   - 模型前向传播处理完整 context
   - Hook 在每层 Attention 后自动压缩 KV Cache
   - 压缩后的 cache 长度从 1000 减少到 500（如果 compression_ratio=0.5）

2. **Decode 阶段**:
   - 基于压缩后的 cache 生成答案
   - 使用贪心解码（greedy decoding）逐字生成
   - 可以回答多个问题（共享同一个压缩后的 cache）

**关键点**:
- `with press(self.model)` 内部调用 `press.__call__(model)`，注册 Hook
- Hook 会在**每层 Attention 计算完成后立即触发**，执行压缩
- 压缩只在 Prefill 阶段执行一次，Decode 阶段复用压缩后的结果

---

## 第三部分：Hook 注册与触发 (`kvpress/presses/base_press.py`)

### 3️⃣ **Hook 注册** - `BasePress.__call__(model)`
```python
@contextmanager
def __call__(self, model):
    """上下文管理器：注册 Hook → 执行 → 移除 Hook"""
    hooks = []
    
    # 遍历模型的所有 Attention 层
    for name, module in model.named_modules():
        if self._is_attention(module):
            # 为每层注册 forward_hook
            hook = module.register_forward_hook(
                self.forward_hook,  # Hook 函数（会在 Attention 后调用）
                with_kwargs=True
            )
            hooks.append(hook)
    
    try:
        yield  # 执行 with 代码块中的代码（model.forward()）
    finally:
        # 执行完毕后移除所有 Hook
        for hook in hooks:
            hook.remove()
```
**作用**: 
- 为模型的所有 Attention 层注册 Hook
- 每层 Attention 计算完成后，自动调用 `forward_hook` 函数
- 执行完毕后自动清理 Hook（避免影响后续推理）

**关键**: 这里的 `yield` 之前注册 Hook，`yield` 之后移除 Hook，确保只在 Prefill 阶段生效。

---

### 4️⃣ **Hook 拦截层** - `BasePress.forward_hook()`
```python
def forward_hook(self, module, input, kwargs, output):
    """每层 Attention 计算完成后自动调用"""
    
    # 判断是否是 Prefill 阶段（只在 Prefill 时压缩）
    cache_position = kwargs.get("cache_position")
    q_len = input[0].shape[1]  # 查询长度
    if cache_position is not None and cache_position[-1] >= q_len:
        return output  # Decode 阶段，跳过压缩
    
    # 提取当前层的 KV Cache
    cache = kwargs["past_key_values"]
    keys, values = extract_keys_and_values(cache, module.layer_idx)
    
    # 🎯 调用子类的 compress 方法（SnapKV 会调用 ScorerPress.compress）
    keys, values = self.compress(
        module,
        hidden_states=input[0],
        keys=keys,
        values=values,
        attentions=kwargs.get("attentions"),
        kwargs=kwargs,
    )
    
    # 将压缩后的 KV 写回 Cache
    cache.update(keys, values, module.layer_idx)
    
    return output
```
**作用**: 
- **拦截每层 Attention 的输出**
- 提取该层完整的 Keys 和 Values
- 调用 `compress` 方法压缩 KV
- 用压缩后的 KV 替换 Cache 中的原始值

**触发时机**: 
- ✅ **Prefill 阶段**: 每层都会触发，执行压缩
- ❌ **Decode 阶段**: 跳过（通过 `cache_position` 判断）

---

## 第四部分：SnapKV 的压缩逻辑

### 5️⃣ **通用压缩层** - `ScorerPress.compress()`
```python
def compress(self, module, hidden_states, keys, values, attentions, kwargs):
    """通用的基于打分的压缩逻辑"""
    
    k_len = keys.shape[2]  # 当前层的序列长度（如 1000）
    
    # 计算保留的 token 数量
    n_kept = int(k_len * (1 - self.compression_ratio))  # 如 1000 * 0.5 = 500
    
    # 🎯 调用子类的 score 方法（SnapKV 实现）
    scores = self.score(
        module=module,
        hidden_states=hidden_states,
        keys=keys,
        values=values,
        attentions=attentions,
        kwargs=kwargs,
    )
    # scores 形状: [batch_size, num_kv_heads, seq_len]
    
    # 选择分数最高的 top-k 个 token
    indices = scores.topk(n_kept, dim=-1).indices  # [batch, num_kv_heads, n_kept]
    indices = indices.sort(dim=-1).values  # 保持原始顺序
    
    # 根据 indices 提取重要的 KV
    keys = keys.gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1]))
    values = values.gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]))
    
    return keys, values  # 返回压缩后的 KV（长度从 1000 → 500）
```
**作用**: 
- 根据 `compression_ratio` 计算需要保留多少 token
- 调用 `score` 函数为每个 token 打分
- 选择分数最高的 top-k 个 token
- 提取对应的 Keys 和 Values

---

### 6️⃣ **🎯 SnapKV 核心算法** - `SnapKVPress.score()`
```python
def score(self, module, hidden_states, keys, values, attentions, kwargs):
    """
    SnapKV 的核心思想：
    用最后 window_size 个 token 的注意力分布作为重要性指标
    """
    
    bsz, num_heads, q_len, head_dim = keys.shape
    window_size = self.window_size  # 默认 64
    
    # ========== Step 1: 计算观察窗口的 Attention ==========
    # 只用最后 window_size 个 token 作为 query
    query_states = hidden_states[:, -window_size:, :]  # [bsz, window_size, hidden_dim]
    
    # 投影到 query 空间
    query_states = module.q_proj(query_states)
    query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)
    
    # 计算注意力分数: Q @ K^T
    attn_weights = torch.matmul(query_states, keys.transpose(2, 3)) / math.sqrt(head_dim)
    # attn_weights: [bsz, num_heads, window_size, q_len]
    
    # Softmax 归一化
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # ========== Step 2: 聚合注意力分数 ==========
    # 对 window_size 维度求平均（每个 token 被观察的平均注意力）
    scores = attn_weights.mean(dim=2)  # [bsz, num_heads, q_len]
    
    return scores
```

**SnapKV 核心思想**:
- **假设**: 如果一个 token 被最后几个 token 关注得多，说明它很重要
- **实现**: 
  1. 用最后 64 个 token 作为 query
  2. 计算它们对所有 token 的注意力权重
  3. 对 64 个 query 的注意力求平均
  4. 得到每个 token 的重要性分数

**为什么只用最后 64 个 token？**
- 计算效率：避免计算完整的 attention（1000x1000）
- 局部性假设：最后几个 token 的注意力分布能代表全局重要性

---

## 🔍 常见疑问解答

### Q1: 是否每层都压缩？
**✅ 是的**。SnapKV **会在模型的每一层**都执行压缩（只要 `compression_ratio > 0`）。

32 层的模型 → `score()` 被调用 32 次（每层一次）

### Q2: 有没有第 0 层？
**✅ 有**。`layer_idx` 从 **0 开始编号**，第 0 层就是模型的第一个 Transformer Layer。

索引范围：`0, 1, 2, ..., 31`（共 32 层）

### Q3: Head 维度如何处理？
**并行处理**。`score` 函数接收的 `keys` 张量形状为 `(BSZ, num_kv_heads, Seq_Len, Head_Dim)`，使用矩阵操作**一次性计算所有 Head** 的分数，而非逐 Head 循环调用。

例如：Llama 3.1 8B 有 8 个 KV Head，一次 `score()` 调用就计算了 8 个 Head 的分数。

### Q4: Prefill vs Decode？
| 阶段 | 是否压缩 | 调用 score? | 说明 |
|------|---------|--------------|------|
| **Prefill** | ✅ 是 | ✅ 调用 | 处理完整 context，计算并压缩 KV Cache |
| **Decode** | ❌ 否 | ❌ 不调用 | 逐字生成答案，复用压缩后的 cache |

**判断依据**: 通过 `cache_position` 和 `q_len` 判断是否是 Prefill 阶段：
```python
if cache_position[-1] >= q_len:
    return  # Decode 阶段，跳过压缩
```

### Q5: 为什么每层压缩后的长度相同？
**✅ 所有层压缩后长度相同**，原因：

1. **Prefill 阶段所有层接收相同长度的 hidden states**（如 1000 个 token）
2. **每层独立生成自己的 KV**（长度都是 1000）
3. **所有层使用相同的压缩率**（如 `compression_ratio=0.5`）
4. **因此每层的 `n_kept = 1000 * 0.5 = 500`**（保留 500 个）

| 层级 | Hidden States 输入 | 生成的 KV 长度 | 压缩后 Cache | 存储位置 |
|------|------------------|---------------|-------------|---------|
| Layer 0 | 1000 | 1000 | 500 | `cache.layers[0]` |
| Layer 1 | 1000 | 1000 | 500 | `cache.layers[1]` |
| Layer 2 | 1000 | 1000 | 500 | `cache.layers[2]` |
| ... | 1000 | 1000 | 500 | ... |
| Layer 31 | 1000 | 1000 | 500 | `cache.layers[31]` |

**关键理解**: 各层的 KV Cache 是**独立存储**的，不是从上一层继承。每层都基于完整的 hidden states 生成自己的 KV，然后独立压缩。

---

## 📌 调用链总结

### 精简版调用路径
```
run_evaluation()                           [评估总入口]
  │
  ├─→ _setup_press()                       [初始化 SnapKV，设置压缩率]
  ├─→ _setup_model_pipeline()              [加载模型和 Pipeline]
  ├─→ _load_and_prepare_dataset()          [加载数据集]
  │
  └─→ _run_inference()                     [🔥 核心推理]
        │
        └─→ pipeline(context, press=..., questions=...)
              │
              └─→ _forward()                [Pipeline 前向传播]
                    │
                    ├─→ with press(model):  [注册 Hook 到所有层]
                    │     │
                    │     └─→ BasePress.__call__()
                    │           └─→ register_forward_hook() × N 层
                    │
                    └─→ model.forward()     [模型前向传播]
                          │
                          └─→ for layer in layers:  [遍历 32 层]
                                │
                                └─→ Attention.forward()
                                      │
                                      └─→ [计算完 Attention 后触发 Hook]
                                            │
                                            └─→ BasePress.forward_hook()
                                                  │
                                                  └─→ ScorerPress.compress()
                                                        │
                                                        └─→ SnapKVPress.score() ✅
                                                              │
                                                              └─→ 返回重要性分数
```

### 关键时间点总结

| 时间点 | 调用的函数 | 作用 |
|--------|-----------|------|
| **评估开始** | `run_evaluation()` | 协调整个评估流程 |
| **加载模型** | `_setup_model_pipeline()` | 加载 Llama 模型和 Tokenizer |
| **初始化 Press** | `_setup_press()` | 创建 SnapKVPress 实例，设置压缩率 |
| **开始推理** | `_run_inference()` | 遍历数据集，调用 Pipeline |
| **Prefill 开始** | `_forward()` 中的 `with press(model)` | 注册 Hook 到所有层 |
| **每层 Attention 后** | `forward_hook()` | 提取 KV，调用 compress |
| **计算重要性** | `score()` | SnapKV 的核心算法（每层调用 1 次）|
| **压缩 KV** | `compress()` | 保留 top-k 重要的 token |
| **Prefill 结束** | Hook 自动卸载 | 移除所有 forward_hook |
| **Decode 开始** | `generate_answer()` | 基于压缩后的 cache 生成答案 |
| **保存结果** | `_save_results()` | 保存预测答案和压缩率 |
| **计算指标** | `_calculate_and_save_metrics()` | 评估模型性能 |

---

## 🎓 技术要点总结

### 1. **KV Cache vs Hidden States**

| 概念 | 作用 | 传递方向 | 是否被压缩 |
|------|------|---------|-----------|
| **Hidden States** | 层与层之间传递的激活值 | Layer N → Layer N+1 | ❌ 不压缩，保持完整 |
| **KV Cache** | 存储各层的 Key/Value，用于 Decode 加速 | 存储在各层内部 | ✅ 每层独立压缩 |

### 2. **压缩时序**

```python
# ❌ 错误理解（边计算边压缩）
Attention 计算中... 
  → 一边生成 KV，一边判断重要性
  → 只把重要的 Token 写入 Cache

# ✅ 正确流程（先完整后压缩）
Step 1: Attention 完整计算
  → 基于完整 hidden states 生成完整的 Keys 和 Values
  → 写入该层的 Cache (包含所有 Token)

Step 2: Hook 立即触发
  → 从 Cache 中读取刚写入的完整 KV
  → 调用 score 函数计算重要性
  → 挑选重要的 Token，丢弃不重要的

Step 3: 覆盖写回 Cache
  → 用压缩后的 KV 替换完整 KV
  → 该层 Cache 现在只包含重要的 Token

Step 4: 传递给下一层
  → 下一层接收完整长度的 hidden states
  → 下一层重复 Step 1-3
```

### 3. **为什么要压缩 KV Cache？**

| 优势 | 说明 |
|------|------|
| **减少显存占用** | 每层只存储重要的 token（如 500 个而非 1000 个）|
| **加速 Decode** | Decode 时只需处理压缩后的 cache（计算量减半）|
| **不影响 Prefill 精度** | Prefill 时仍然使用完整序列，只压缩存储的 cache |
| **支持超长上下文** | 可以在有限显存下处理更长的文本（如 128K tokens）|

### 4. **SnapKV 的核心假设**

> **"如果一个 token 被最后几个 token 关注得多，说明它很重要"**

- 用最后 64 个 token 作为"观察窗口"
- 计算它们对所有历史 token 的注意力权重
- 注意力权重高 → 该 token 重要 → 保留到 cache 中
- 注意力权重低 → 该 token 不重要 → 丢弃

### 5. **代码证据**

来自 `base_press.py` 第 142-154 行：
```python
def forward_hook(self, module, input, kwargs, output):
    # 此时 Attention 已经完成计算，完整的 KV 已在 cache 中
    cache = kwargs["past_key_values"]
    
    # Step 1: 从 cache 中提取完整的 keys 和 values
    keys, values = extract_keys_and_values(cache, module.layer_idx)
    
    # Step 2: 调用 compress (内部调用 score)，返回压缩后的 KV
    keys, values = self.compress(module, hidden_states, keys, values, ...)
    
    # Step 3: 用压缩后的 KV 覆盖 cache 中的原有数据
    cache.update(keys, values, module.layer_idx)
    
    return output  # 继续传递给下一层
```

---

## 🚀 完整示例：1000 tokens → 500 tokens

假设输入 context 有 1000 个 token，模型有 32 层，`compression_ratio=0.5`：

```
评估开始
  ↓
加载模型: Meta-Llama-3.1-8B-Instruct (32 层)
加载数据集: LongBench-E HotpotQA (只用 1% 数据快速测试)
初始化 SnapKV: compression_ratio=0.5, window_size=64
  ↓
开始推理第 1 个 context (1000 tokens)
  ↓
Tokenize: 转换为 token IDs [101, 234, 567, ..., 999] (1000 个)
  ↓
========== Prefill 阶段 ==========
注册 Hook 到 32 层
  ↓
Layer 0:
  - 输入: hidden_states (1000 tokens)
  - 生成: keys, values (1000 tokens)
  - Hook 触发 → score() 计算 → 保留 500 个重要 tokens
  - cache.layers[0] 存储: 500 tokens ✅
  ↓
Layer 1:
  - 输入: hidden_states (1000 tokens) ← 仍然是完整的！
  - 生成: keys, values (1000 tokens)
  - Hook 触发 → score() 计算 → 保留 500 个重要 tokens
  - cache.layers[1] 存储: 500 tokens ✅
  ↓
... Layer 2 ~ 31 重复相同流程 ...
  ↓
Prefill 完成，移除所有 Hook
每层 cache 都只存储 500 个 tokens（总共节省了 50% 显存）
  ↓
========== Decode 阶段 ==========
拼接问题: "What is the capital of France?"
  ↓
生成答案（逐字生成）:
  Token 1: "The"   ← 使用压缩后的 cache (500 tokens)
  Token 2: "capital"
  Token 3: "of"
  Token 4: "France"
  Token 5: "is"
  Token 6: "Paris"
  Token 7: "."
  ↓
答案生成完毕: "The capital of France is Paris."
  ↓
保存结果到 CSV 和 JSON
评估完成 ✅
```

---

## 📖 相关文件索引

| 文件路径 | 关键函数 | 作用 |
|---------|---------|------|
| `evaluation/evaluate.py` | `run_evaluation()` | 评估总入口，协调所有子流程 |
| `evaluation/evaluate.py` | `_run_inference()` | 遍历数据集，调用 Pipeline 推理 |
| `kvpress/pipeline.py` | `_forward()` | Pipeline 核心，触发 Prefill 和 Decode |
| `kvpress/presses/base_press.py` | `__call__()` | 注册 Hook 到所有层 |
| `kvpress/presses/base_press.py` | `forward_hook()` | 拦截 Attention 输出，调用压缩 |
| `kvpress/presses/scorer_press.py` | `compress()` | 通用压缩逻辑，调用子类 score |
| `kvpress/presses/snapkv_press.py` | `score()` | SnapKV 核心算法，计算重要性 |
| `evaluation/debug_config.yaml` | 配置文件 | 设置模型、数据集、压缩率等参数 |

---

## 🎯 核心结论

1. **`run_evaluation()` 是评估的总入口**，按顺序调用 9 个子函数完成评估
2. **`_run_inference()` 触发推理**，将 SnapKV Press 传入 Pipeline
3. **`_forward()` 分两阶段**：Prefill（压缩 KV）+ Decode（生成答案）
4. **`with press(model)` 注册 Hook**，在每层 Attention 后自动调用压缩
5. **`score()` 被调用 N 次**（N = 模型层数），每层独立计算重要性
6. **每层压缩后长度相同**，因为输入长度、压缩率都相同
7. **KV Cache 每层独立**，不同层可以保留不同的重要 token
8. **压缩只在 Prefill 阶段**，Decode 阶段复用压缩后的 cache

---

## 💡 扩展阅读

- **其他 Press 方法**: KnormPress, ExpectedAttentionPress, BlockPress 等
- **多问题处理**: 同一个 context 如何复用压缩后的 cache 回答多个问题
- **Decoding 压缩**: DecodingPress 在生成阶段动态压缩 cache
- **性能分析**: 压缩率对准确率和速度的影响

---

*文档生成时间: 2025-02-06*  
*基于 KVPress 代码版本: 最新*

---

## ⏱️ Prefill 阶段的精确时序

### 🔑 核心概念：KV Cache vs Hidden States

**必须理解的关键点**：

| 概念 | 作用 | 传递方向 | 是否被压缩 |
|------|------|---------|-----------|
| **Hidden States** | 层与层之间传递的激活值 | Layer N → Layer N+1 | ❌ 不压缩，保持完整 |
| **KV Cache** | 存储各层的 Key/Value，仅用于 Decode 加速 | 存储在各层内部 | ✅ 每层独立压缩 |

**错误理解** ❌：  
"Layer 0 压缩后只有 500 个 token，所以 Layer 1 只能看到 500 个 token"

**正确理解** ✅：  
- Layer 0 压缩的是**自己的 KV Cache**（存储下来给 Decode 用）
- Layer 1 接收的是 Layer 0 的 **Hidden States 输出**（保持完整 1000 个）
- Layer 1 基于完整的 1000 个 hidden states 生成自己的 KV，然后独立压缩

---

### 重要澄清：先完整计算，再立即压缩

**你的理解核心正确，但时序上有细微差别：**

#### ❌ 错误理解（边计算边压缩）
```
Attention 计算中... 
  → 一边生成 KV，一边判断重要性
  → 只把重要的 Token 写入 Cache
```

#### ✅ 正确流程（先完整后压缩）
```
Step 1: Attention 完整计算（当前层）
  → 模型基于完整 hidden states 生成完整的 Keys 和 Values
  → 写入该层的 Cache (此时该层 Cache 包含所有 Token)

Step 2: Hook 立即触发（当前层）
  → 从该层 Cache 中读取刚写入的完整 KV
  → 调用 score 函数计算重要性
  → 挑选重要的 Token，丢弃不重要的

Step 3: 覆盖写回 Cache（当前层）
  → 用压缩后的 KV 替换该层原有的完整 KV
  → 该层 Cache 现在只包含重要的 Token

Step 4: 传递给下一层
  → 下一层接收完整长度的 hidden states（不是 KV Cache！）
  → 下一层重复 Step 1-3，生成自己的完整 KV 并独立压缩
  → 每层的 KV Cache 是独立的，互不影响
```

---

### 📍 代码证据 (`base_press.py` 第 142-154 行)

```python
def forward_hook(self, module, input, kwargs, output):
    # 此时 Attention 已经完成计算，完整的 KV 已在 cache 中
    cache = kwargs["past_key_values"]
    
    # Step 1: 从 cache 中提取完整的 keys 和 values
    keys, values = extract_keys_and_values(cache, module.layer_idx)
    
    # Step 2: 调用 compress (内部调用 score)，返回压缩后的 KV
    keys, values = self.compress(module, hidden_states, keys, values, ...)
    
    # Step 3: 用压缩后的 KV 覆盖 cache 中的原有数据
    cache_layer.keys = keys
    cache_layer.values = values
    
    return output  # 继续传递给下一层
```

---

### 🔄 每层独立压缩机制

假设输入序列有 1000 个 Token，压缩率 50%：

| 层级 | 输入 Hidden States 长度 | 该层生成的 KV 长度 | 压缩后 Cache 长度 | 
|------|------------------------|-------------------|-------------------|
| **Layer 0** | 1000 | 1000 | 500 |
| **Layer 1** | 1000 | 1000 | 500 |
| **Layer 2** | 1000 | 1000 | 500 |
| ... | 1000 | 1000 | 500 |
| **Layer 31** | 1000 | 1000 | 500 |

**关键观察**：
- ✅ **每层的 KV Cache 是独立的**（不是从上一层继承）
- ✅ 在 Prefill 阶段，**所有层接收相同长度的 hidden states**（1000 个 token）
- ✅ **每层都生成完整长度的 KV**（1000 个），然后独立压缩到相同比例（500 个）
- ✅ **压缩比例对所有层相同**，所以每层保留的 token 数量也相同
- ⚠️ KV Cache 只在 **Decode 阶段**被使用，Prefill 阶段各层间传递的是 hidden states

---

### 🎯 回答你的问题

> **是不是一边 prefill 一边计算当前层哪些 token 比较重要？**

**准确答案**：  
不是"一边 Attention 计算一边压缩"，而是：
1. **当前层先完整计算 Attention**（所有 Token 都参与）
2. **计算完成后立即触发 Hook**
3. **Hook 中调用 score 判断重要性并压缩该层的 KV**
4. **压缩结果写回该层 Cache，替换完整版本**
5. **下一层接收完整的 hidden states**（不受上一层压缩影响）
6. **下一层重复 1-4 步骤**，独立生成和压缩自己的 KV

> **只将重要的 token 写入 kv 缓存吗？**

**准确答案**：  
不是"只写入重要的"，而是：
1. **先写入所有 Token 的 KV**（Attention 层的正常行为）
2. **然后立即用重要的 Token 覆盖掉完整的 KV**（Hook 的作用）
3. **最终该层 Cache 中只剩下重要的 Token**
4. **每层都是独立的**：每层都生成完整 KV → 压缩 → 存储压缩版本

---

### 💡 为什么要这样设计？

**技术原因**：
- **Prefill 阶段**：每层都需要访问完整的输入序列，所以 hidden states 保持完整长度
- **Decode 阶段**：生成新 token 时，需要使用历史 KV Cache 来加速计算
- 压缩 KV Cache 可以：
  - ✅ **减少显存占用**（每层只存储重要的 KV）
  - ✅ **加速 Decode**（Decode 时只需要处理压缩后的 Cache）
  - ✅ **不影响 Prefill 精度**（Prefill 时仍然使用完整序列）

**关键理解**：
- **KV Cache 的作用时机**：只在 Decode 阶段使用，避免重复计算历史 token 的 KV
- **为什么每层独立压缩**：因为每层的 KV 都是独立生成的，不同层的重要 token 可能不同
- **压缩不影响前向传播**：各层间传递的是 hidden states（保持完整），而不是 KV Cache

---

## 🧪 代码验证：为什么每层压缩长度相同

### 关键代码路径

#### 1. Hook 提取当前层的 KV（`base_press.py:143`）
```python
def forward_hook(self, module, input, kwargs, output):
    # module.layer_idx 是当前层的索引（0, 1, 2, ...）
    keys, values = extract_keys_and_values(cache, module.layer_idx)
    # ↑ 提取的是当前层自己的 KV，不是上一层的！
```

#### 2. 提取函数（`utils.py:104`）
```python
def extract_keys_and_values(cache: Cache, layer_idx: int):
    # cache.layers 是一个列表，每层有独立的 cache
    keys = cache.layers[layer_idx].keys  # 当前层的 keys
    values = cache.layers[layer_idx].values  # 当前层的 values
    return keys, values
```

#### 3. 压缩计算（`scorer_press.py:90`）
```python
def compress(self, module, hidden_states, keys, values, attentions, kwargs):
    # keys.shape = (batch, num_kv_heads, seq_len, head_dim)
    k_len = keys.shape[2]  # 当前层的序列长度
    
    # 计算保留数量（所有层使用相同的 compression_ratio）
    n_kept = int(k_len * (1 - self.compression_ratio))
    # ↑ 如果 k_len 对每层相同，n_kept 也对每层相同
    
    scores = self.score(...)
    indices = scores.topk(n_kept, dim=-1).indices
    # 选择 top-k 个重要的 token
```

### 逻辑推导

**Prefill 阶段（输入 1000 个 token，compression_ratio=0.5）**：

```python
# Layer 0
hidden_states_0 = input_embeddings  # shape: (batch, 1000, hidden_dim)
keys_0, values_0 = Attention_0(hidden_states_0)  # shape: (batch, num_kv_heads, 1000, head_dim)
# Hook 触发
k_len = 1000  # keys_0.shape[2]
n_kept = int(1000 * (1 - 0.5)) = 500
compressed_keys_0 = keys_0[..., top_500_indices, :]  # 压缩到 500
cache.layers[0].keys = compressed_keys_0  # 存储在 Layer 0 的 cache 中

# Layer 1
hidden_states_1 = output_from_layer_0  # shape: (batch, 1000, hidden_dim) ← 仍然是 1000！
keys_1, values_1 = Attention_1(hidden_states_1)  # shape: (batch, num_kv_heads, 1000, head_dim)
# Hook 触发
k_len = 1000  # keys_1.shape[2] ← 仍然是 1000！
n_kept = int(1000 * (1 - 0.5)) = 500  # ← 仍然保留 500！
compressed_keys_1 = keys_1[..., top_500_indices, :]  # 压缩到 500
cache.layers[1].keys = compressed_keys_1  # 存储在 Layer 1 的 cache 中

# Layer 2, 3, ... 31：重复上述过程
# 每层都生成 1000 个 KV，压缩到 500 个，存储在各自的 cache 中
```

### 正确的层级表格

| 层级 | Hidden States 输入 | 生成的 KV 长度 | 压缩后 Cache | 存储位置 |
|------|------------------|---------------|-------------|---------|
| Layer 0 | 1000 | 1000 | 500 | `cache.layers[0]` |
| Layer 1 | 1000 | 1000 | 500 | `cache.layers[1]` |
| Layer 2 | 1000 | 1000 | 500 | `cache.layers[2]` |
| ... | 1000 | 1000 | 500 | ... |
| Layer 31 | 1000 | 1000 | 500 | `cache.layers[31]` |

### 结论

✅ **每层压缩后的长度完全相同**，因为：
1. Prefill 阶段所有层的 hidden states 输入长度相同（1000）
2. 每层独立生成自己的 KV，长度相同（1000）
3. 压缩比例对所有层相同（`compression_ratio=0.5`）
4. 因此每层的 `n_kept` 相同（500）

❌ **"逐层递减"的理解是错误的**，那混淆了：
- **KV Cache**（各层独立存储，互不影响）
- **Hidden States**（层间传递，保持完整长度）
