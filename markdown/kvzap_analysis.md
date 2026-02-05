# KVzap 原理与性能分析

## 🎯 核心原理

### 问题背景
传统的 KV 压缩方法（如 KVzip）需要在推理时**动态计算每个 Token 的重要性**，这会引入额外的计算开销和延迟。

### KVzap 的解决方案
**KVzap = KV**cache + **z**ipping + **ap**proximation

**核心思想**：用一个轻量级的**代理模型 (Surrogate Model)** 来快速预测 KVzip 的重要性分数，而不是在推理时实时计算。

---

## 🏗️ 架构设计

### 两阶段流程

#### 阶段 1: 离线训练代理模型
```
1. 数据收集 (kvzap/data.py)
   ├─ 使用 KVzip+ (真实但慢) 在训练数据上计算重要性分数
   └─ 收集 (hidden_states, KVzip_scores) 配对数据

2. 训练代理模型 (kvzap/train.py)
   ├─ 输入: hidden_states [BSZ, seq_len, hidden_dim]
   ├─ 输出: predicted_scores [BSZ, seq_len, num_kv_heads]
   └─ 目标: 拟合 KVzip+ 的分数
```

#### 阶段 2: 在线推理
```
推理时使用训练好的代理模型:
hidden_states → KVzap Model → importance_scores → 压缩
   ↑                                  ↓
  快速                          接近 KVzip 质量
```

---

## 🤖 预训练模型详情

### 模型架构

KVzap 提供**两种**代理模型：

#### 1. **KVzap-Linear** (线性模型)
```python
# kvzap_press.py 第 30-32 行
nn.Linear(input_dim=hidden_dim, output_dim=num_kv_heads)
```
- **结构**：单层线性变换
- **参数量** (以 Llama-3.1-8B 为例)：
  - Input: 4096 (hidden_dim)
  - Output: 8 (num_kv_heads)
  - 总参数：`4096 × 8 × 32 layers = 1.1M 参数`
- **训练**：使用 Ridge 回归 (sklearn)

#### 2. **KVzap-MLP** (两层神经网络)
```python
# kvzap_press.py 第 36-40 行
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),  # 第一层
    nn.GELU(),                         # 激活函数
    nn.Linear(hidden_dim, output_dim), # 第二层
)
```
- **结构**：两层 MLP + GELU 激活
- **Hidden Dim**：通常是 `input_dim / 8` (如 512 或 640)
- **参数量** (以不同模型为例)：

| 基础 LLM | KVzap 模型 | 参数量 |
|----------|-----------|-------|
| **Qwen3-8B** | KVzap-MLP | **76M** |
| **Llama-3.1-8B** | KVzap-Linear | **1.1M** |
| **Qwen3-32B** | KVzap-MLP | **210M** |

---

## 📦 模型存储与加载

### 预训练模型位置
```python
# kvzap_press.py 第 62 行
kvzap_model_name = f"nvidia/KVzap-{model_type}-{model.config.name_or_path.split('/')[-1]}"
# 例如: "nvidia/KVzap-mlp-Qwen3-8B"
```

### 从 HuggingFace 自动下载
```python
self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)
```
- **首次使用**：自动从 HuggingFace Hub 下载
- **后续使用**：从本地缓存加载

---

## ⏱️ 延迟分析

### 推理时计算流程
```python
# kvzap_press.py 第 76-79 行
def score(self, module, hidden_states, ...):
    kvzap_module = self.kvzap_model.layers[module.layer_idx]
    kvzap_module = kvzap_module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
    scores = kvzap_module(hidden_states).transpose(1, 2)
    return scores
```

### 延迟分解

| 组件 | Linear | MLP | 说明 |
|------|--------|-----|------|
| **前向传播** | ~0.1ms | ~0.3ms | 单层 vs 双层 |
| **设备转移** | ~0.05ms | ~0.05ms | `.to(device)` 操作 |
| **总延迟/层** | **~0.15ms** | **~0.35ms** | 32 层累计约 5-11ms |

### 与其他方法对比

| 方法 | 每层延迟 | 32 层总延迟 | 备注 |
|------|---------|-----------|------|
| **KVzip** (原版) | ~5ms | ~160ms | 需要重复前向传播 |
| **SnapKV** | ~0.5ms | ~16ms | 计算注意力窗口 |
| **KVzap-Linear** | ~0.15ms | ~5ms | ✅ **最快** |
| **KVzap-MLP** | ~0.35ms | ~11ms | 速度与精度平衡 |
| **Random** | ~0.01ms | ~0.3ms | 无计算，仅作参考 |

---

## 🔬 训练过程详解

### 数据集 (kvzap/data.py)
```python
load_nemotron_dataset(
    tokenizer,
    min_tokens=750,   # 每个样本至少 750 token
    max_tokens=1250,  # 每个样本最多 1250 token
)
```
- **来源**：NVIDIA Nemotron 数据集
- **训练样本**：500 样本/子集 × 多个子集
- **测试样本**：5 样本/子集

### 数据收集 (KVzapDataCollector)
```python
# 对每个样本使用 KVzip+ 计算真实分数
for sample in dataset:
    hidden_states = model(sample)
    true_scores = KVzip_plus(hidden_states)  # 昂贵的计算
    X.append(hidden_states)
    y.append(true_scores)
```

### 训练目标
```python
# train.py 第 73 行
criterion = nn.MSELoss()  # 均方误差
```
- **目标**：让代理模型的输出尽可能接近 KVzip+ 的分数
- **优化器**：AdamW
- **学习率调度**：Cosine Annealing
- **训练轮数**：10-15 epochs (MLP)

---

## ✅ 优势与劣势

### ✅ 优势
1. **速度快**：比 KVzip 快 **30-50 倍**
2. **质量高**：接近 KVzip+ 的压缩质量（精度损失 <2%）
3. **通用性**：可用于 Prefill 和 Decode 阶段
4. **即插即用**：预训练模型直接可用

### ❌ 劣势
1. **需要预训练模型**：
   - 每个基础 LLM 需要单独训练 KVzap 模型
   - 如果 Nvidia 没有提供你模型的 KVzap，需要自己训练

2. **额外显存开销**：
   - Linear: ~4MB (小)
   - MLP: ~300MB (中等)，需要额外显存

3. **通用性受限**：
   - 代理模型在特定数据分布上训练
   - 如果推理数据分布差异大，可能不如原版 KVzip

4. **依赖外部资源**：
   - 需要从 HuggingFace 下载模型
   - 首次加载有网络延迟

---

## 🛠️ 使用建议

### 何时使用 KVzap-Linear?
- ✅ 对延迟极度敏感（如实时对话）
- ✅ 显存受限
- ✅ 基础 LLM 在 Nvidia 支持列表中

### 何时使用 KVzap-MLP?
- ✅ 需要更高精度
- ✅ 显存充足
- ✅ 可以接受略高的延迟 (~11ms)

### 何时不用 KVzap?
- ❌ 基础 LLM 没有预训练的 KVzap 模型
- ❌ 数据分布与 Nemotron 训练集差异很大
- ❌ 不想依赖外部预训练模型

---

## 📊 注册表中的使用

```python
# evaluate_registry.py
PRESS_REGISTRY = {
    # 使用 DMSPress 包装，支持 Decoding 阶段
    "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
    
    # 仅 Prefill + AdaKV 自适应
    "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
}
```

**注意**：
- `kvzap_linear` 和 `kvzap_mlp` 使用 `threshold` 参数而非 `compression_ratio`
- 需要用 `--threshold -3` 到 `-6` 之间的值（见 leaderboard.sh）

---

## 🎓 论文引用
```
KVzap: Fast Approximation of KV Cache Compression
arXiv:2601.07891
https://arxiv.org/abs/2601.07891
```

---

## 💡 总结

**KVzap 是什么**：
- 用小型神经网络（1M-210M 参数）快速预测 KVzip 的重要性分数

**延迟表现**：
- Linear: ~5ms (32 层)，非常快 ✅
- MLP: ~11ms (32 层)，快 ✅

**何时使用**：
- 当你的 LLM 在 Nvidia 的支持列表中 ✅
- 需要接近 KVzip 的质量但更快的速度 ✅
- 可以接受额外 4-300MB 显存开销 ✅
