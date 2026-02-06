# SnapKV 调用链详解

## 🎯 核心问题
**何时进入 `SnapKVPress.score` 函数？**

---

## 📊 调用链

### 1️⃣ **启动层** - `evaluate.py`
```python
EvaluationRunner._run_inference()
  │
  └─→ self.pipeline(context, questions=..., press=self.press, ...)
```
**作用**: 触发推理流程，将 `SnapKVPress` 实例传入 Pipeline。

---

### 2️⃣ **Pipeline 层** - `kvpress/pipeline.py`
```python
KVPressTextGenerationPipeline._forward()
  │
  ├─→ with press(self.model):  # 向所有 Attention 层注册 Hook
  │     │
  │     └─→ BasePress.__call__()  # 注册 forward_hook 到每层
  │
  └─→ self.model.model(input_ids=context_ids, past_key_values=cache)
```
**作用**: 将 SnapKV 的 Hook 安装到模型的所有 Transformer Layer 上。

---

### 3️⃣ **模型层** - `transformers/.../modeling_xxx.py`
```python
LlamaModel.forward()
  │
  └─→ for layer in layers:  # 遍历所有 Decoder Layer (如 32 层)
        │
        └─→ LlamaDecoderLayer.forward()
              │
              └─→ LlamaAttention.forward()
                    │
                    └─→ [完成 KV 计算后触发 Hook]
```
**作用**: 逐层前向传播，每层计算完 Attention 后触发 Hook。

---

### 4️⃣ **Hook 拦截层** - `kvpress/presses/base_press.py`
```python
BasePress.forward_hook()  # ⚠️ 每层都会调用
  │
  ├─→ if cache_position[-1] > q_len: return  # 跳过 Decode 阶段
  │
  ├─→ keys, values = extract_keys_and_values(cache, module.layer_idx)
  │
  └─→ self.compress(module, hidden_states, keys, values, ...)
```
**触发时机**: **仅在 Prefill 阶段**，每层计算完 Attention 后立即调用。  
**层级范围**: 所有层 (`layer_idx` 从 0 到 N-1)。

---

### 5️⃣ **通用压缩层** - `kvpress/presses/scorer_press.py`
```python
ScorerPress.compress()
  │
  ├─→ if self.compression_ratio == 0: return keys, values  # 不压缩则跳过
  │
  ├─→ scores = self.score(...)  # 🎯 调用子类的 score 函数
  │
  └─→ indices = scores.topk(n_kept).indices  # 根据 score 挑选 Token
```
**作用**: 通用逻辑，调用子类的 `score` 方法计算重要性。

---

### 6️⃣ **🎯 目标函数** - `kvpress/presses/snapkv_press.py`
```python
SnapKVPress.score(module, hidden_states, keys, values, attentions, kwargs)
  │
  └─→ 计算最后 window_size 个 Token 对前面所有 Token 的注意力
      │
      └─→ 返回 scores: [BSZ, num_kv_heads, seq_len]
```
**执行频率**: 如果模型有 32 层，**调用 32 次**（每层一次）。  
**Head 处理**: 一次性并行计算该层 **所有 KV Head** 的分数，而非每个 Head 单独调用。

---

## 🔍 关键细节

### Q1: 是否每层都压缩？
**是的**。SnapKV **会在模型的每一层**都执行压缩（只要 `compression_ratio > 0`）。

### Q2: 有没有第 0 层？
**有**。`layer_idx` 从 **0 开始编号**，第 0 层就是模型的第一个 Transformer Layer。

### Q3: Head 维度如何处理？
**并行处理**。`score` 函数接收的 `keys` 张量形状为 `(BSZ, num_kv_heads, Seq_Len, Head_Dim)`，使用矩阵操作**一次性计算所有 Head** 的分数，而非逐 Head 循环调用。

### Q4: Prefill vs Decode？
- **Prefill 阶段** (处理 Context)：✅ 执行压缩，调用 `score`
- **Decode 阶段** (逐字生成)：❌ 跳过压缩，不调用 `score`

---

## 📌 总结

**调用路径精简版**:
```
evaluate.py → pipeline._forward() → with press(model) → model.forward()
  → [每层] Attention.forward() → Hook 拦截 → scorer_press.compress()
  → SnapKVPress.score() ✅ [返回重要性分数]
```

**关键点**:  
- **调用次数** = 模型层数（如 32 层 = 32 次调用）
- **处理维度**: 每次处理该层所有 KV Head 的数据（并行计算）
- **生效阶段**: 仅 Prefill，Decode 阶段不执行

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
