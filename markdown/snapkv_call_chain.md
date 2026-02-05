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
Step 1: Attention 完整计算
  → 模型为所有 Token 生成完整的 Keys 和 Values
  → 写入 Cache (此时 Cache 包含所有 Token)

Step 2: Hook 立即触发 (在同一层内)
  → 从 Cache 中读取刚写入的完整 KV
  → 调用 score 函数计算重要性
  → 挑选重要的 Token，丢弃不重要的

Step 3: 覆盖写回 Cache
  → 用压缩后的 KV 替换原有的完整 KV
  → Cache 现在只包含重要的 Token

Step 4: 传递给下一层
  → 下一层读取的是已压缩的 Cache
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

### 🔄 逐层压缩的波动效应

假设输入序列有 1000 个 Token，压缩率 50%：

| 层级 | 输入 Cache 长度 | 计算中的 KV 长度 | 压缩后 Cache 长度 | 传递给下一层 |
|------|----------------|-----------------|----------------|-------------|
| **Layer 0** | 0 (空) | 1000 | 500 | 500 |
| **Layer 1** | 500 | 500 | 250 | 250 |
| **Layer 2** | 250 | 250 | 125 | 125 |
| ... | ... | ... | ... | ... |
| **Layer 31** | ~1-2 | ~1-2 | ~1 | ~1 |

**关键观察**：
- ✅ 每层都执行**完整的 Attention 计算**（使用当前层输入的完整序列）
- ✅ 计算完成后**立即压缩**该层的 KV Cache
- ✅ 压缩后的 Cache 作为**下一层的上下文**
- ⚠️ 越深的层，接收到的上下文越短（因为被前面层压缩过了）

---

### 🎯 回答你的问题

> **是不是一边 prefill 一边计算当前层哪些 token 比较重要？**

**准确答案**：  
不是"一边 Attention 计算一边压缩"，而是：
1. **当前层先完整计算 Attention**（所有 Token 都参与）
2. **计算完成后立即触发 Hook**
3. **Hook 中调用 score 判断重要性并压缩**
4. **压缩结果写回 Cache，替换完整版本**
5. **下一层使用的是压缩后的 Cache**

> **只将重要的 token 写入 kv 缓存吗？**

**准确答案**：  
不是"只写入重要的"，而是：
1. **先写入所有 Token 的 KV**（Attention 层的正常行为）
2. **然后立即用重要的 Token 覆盖掉完整的 KV**（Hook 的作用）
3. **最终 Cache 中只剩下重要的 Token**（对下游层来说，就像从未有过不重要的 Token）

---

### 💡 为什么要这样设计？

**技术原因**：
- Transformer 的 Attention 机制需要"看到"所有 Token 才能计算当前层的输出
- 但后续层不需要保留所有历史 Token，只需保留重要的部分
- 通过 Hook 机制，在不修改模型源码的情况下实现压缩

**性能优化**：
- 当前层：计算完整，精度不损失
- 后续层：只处理压缩后的 Cache，速度更快，内存更少
