# KVPress 类继承关系分析

## 一、继承体系概述

KVPress 的类继承体系分为三个层次：

```
BasePress (基类)
├── ScorerPress (基于评分的基类)
│   ├── [多个具体的评分压缩方法]
│   └── ...
└── [复杂逻辑的压缩方法]
```

## 二、核心区别

### BasePress
- **定义**：所有 KV cache 压缩方法的基类
- **核心方法**：`compress(module, hidden_states, keys, values, attentions, kwargs)`
- **特点**：需要子类自己实现完整的压缩逻辑

### ScorerPress
- **定义**：基于评分的 KV cache 压缩基类（继承自 BasePress）
- **核心方法**：`score(module, hidden_states, keys, values, attentions, kwargs)` - 返回每个 token 的重要性分数
- **特点**：
  - 已经实现了 `compress` 方法：计算分数 → 选择 top-k → 剪枝
  - 子类只需实现 `score` 方法来定义如何计算重要性
  - 适用于"评分 + 剪枝"这种简单模式的压缩方法

## 三、详细分类

### 1. 直接继承 ScorerPress 的类（基于评分的压缩方法）

这些类只需要实现 `score` 方法来定义如何计算 token 的重要性分数，剪枝逻辑由 ScorerPress 统一处理。

| 类名 | 文件 | 评分依据 | 说明 |
|------|------|----------|------|
| **SnapKVPress** | `snapkv_press.py` | 最近 token 的注意力模式 | 使用最近 window_size 个 token 的注意力权重估计重要性 |
| **KnormPress** | `knorm_press.py` | Key 向量的 L2 范数 | 简单高效，只需计算范数 |
| **RandomPress** | `random_press.py` | 随机分数 | 用于基线对比 |
| **ObservedAttentionPress** | `observed_attention_press.py` | 观察到的注意力权重 | 直接使用计算得到的注意力权重 |
| **ExpectedAttentionPress** | `expected_attention_press.py` | 预期的注意力分数 | 基于 Query-Key 相似度预测注意力 |
| **StreamingLLMPress** | `streaming_llm_press.py` | 保留初始和最近的 token | 固定窗口策略 |
| **TOVAPress** | `tova_press.py` | Token 重要性 | TOVA 方法 |
| **QFilterPress** | `qfilter_press.py` | Query 过滤 | 基于 Query 的过滤机制 |
| **LeverageScorePress** | `leverage_press.py` | 杠杆分数 | 基于统计杠杆的重要性 |
| **LagKVPress** | `lagkv_press.py` | 滞后关系 | LagKV 方法 |
| **KeyDiffPress** | `keydiff_press.py` | Key 差异 | 基于 Key 的差异性 |
| **CriticalKVPress** | `criticalkv_press.py` | 关键 KV 对 | 识别关键的 KV 对 |
| **CompactorPress** | `compactor_press.py` | 紧凑性评分 | Compactor 方法 |
| **CURPress** | `cur_press.py` | CUR 分解 | 基于 CUR 分解的评分 |
| **NonCausalAttnPress** | `non_causal_attention_press.py` | 非因果注意力 | 非因果注意力模式 |
| **KVzapPress** | `kvzap_press.py` | KVzap 评分 | KVzap 压缩方法 |
| **PyramidKVPress** | `pyramidkv_press.py` | 继承自 SnapKVPress | 金字塔式压缩（二级继承） |

### 2. 直接继承 BasePress 的类（复杂逻辑的压缩方法）

这些类有更复杂的压缩逻辑，不是简单的"评分 + 剪枝"模式，需要自己实现完整的 `compress` 方法。

#### 2.1 包装器类（Wrapper Classes）

这些类包装其他 Press 来实现特殊功能：

| 类名 | 文件 | 功能 | 说明 |
|------|------|------|------|
| **ComposedPress** | `composed_press.py` | 组合多个 Press | 顺序应用多个压缩方法 |
| **AdaKVPress** | `adakv_press.py` | 头特异性压缩 | 包装一个 ScorerPress，实现每个注意力头的自适应压缩 |
| **ChunkKVPress** | `chunkkv_press.py` | 分块选择 | 包装一个 ScorerPress，基于块而非全局选择 token |
| **BlockPress** | `block_press.py` | 块状迭代压缩 | 包装一个 ScorerPress，在固定大小的块中迭代压缩 |
| **ChunkPress** | `chunk_press.py` | 分块压缩 | 基于块的压缩策略 |
| **PerLayerCompressionPress** | `per_layer_compression_press.py` | 每层不同压缩率 | 为不同层应用不同的压缩策略 |
| **PrefillDecodingPress** | `prefill_decoding_press.py` | 预填充-解码分离 | 预填充和解码阶段使用不同策略 |

#### 2.2 维度压缩类

这些类压缩的是 key/value 的维度，而不是序列长度：

| 类名 | 文件 | 压缩维度 | 说明 |
|------|------|----------|------|
| **ThinKPress** | `think_press.py` | Key 的通道维度 | 压缩 key 的维度而非序列长度 |
| **KVzipPress** | `kvzip_press.py` | Key-Value 维度 | KVzip 压缩方法 |

#### 2.3 特殊注意力机制类

这些类修改或优化注意力计算方式：

| 类名 | 文件 | 特点 | 说明 |
|------|------|------|------|
| **DuoAttentionPress** | `duo_attention_press.py` | 双重注意力 | 使用预训练的注意力模式 |
| **CriticalAdaKVPress** | `criticalkv_press.py` | 关键 KV + 自适应 | CriticalKV 的自适应变体 |
| **DMSPress** | `dms_press.py` | DMS 方法 | Dynamic Memory Scheduling |
| **FinchPress** | `finch_press.py` | Finch 方法 | Finch 压缩策略 |
| **SimLayerKVPress** | `simlayerkv_press.py` | 层相似性 | 利用层间相似性 |
| **KeyRerotationPress** | `key_rerotation_press.py` | Key 重旋转 | 重新旋转 Key 向量 |
| **DecodingPress** | `decoding_press.py` | 解码优化 | 解码阶段的特殊处理 |

### 3. 二级继承（继承自具体 Press 类）

| 类名 | 父类 | 文件 | 说明 |
|------|------|------|------|
| **PyramidKVPress** | SnapKVPress | `pyramidkv_press.py` | 金字塔式的 SnapKV 变体 |
| **ExpectedAttentionStatsPress** | ExpectedAttentionPress | `expected_attention_with_stats.py` | 带统计信息的预期注意力 |

## 四、设计模式分析

### 为什么需要 ScorerPress？

1. **代码复用**：大多数压缩方法的流程是相同的：
   - 计算每个 token 的重要性分数
   - 根据 `compression_ratio` 选择 top-k
   - 使用 `gather` 剪枝 keys 和 values

2. **简化实现**：使用 ScorerPress，新的压缩方法只需：
   ```python
   @dataclass
   class MyPress(ScorerPress):
       def score(self, module, hidden_states, keys, values, attentions, kwargs):
           # 只需实现如何计算分数
           return scores  # shape: (batch_size, num_kv_heads, seq_len)
   ```

3. **一致性**：确保所有基于评分的方法具有相同的压缩行为

### 为什么有些类直接继承 BasePress？

当压缩逻辑**不符合**"评分 + top-k 剪枝"模式时，需要直接继承 BasePress：

1. **AdaKVPress**：虽然使用评分，但采用头特异性剪枝，并且使用掩码而非直接剪枝
2. **ThinKPress**：压缩的是维度，而不是序列长度
3. **ChunkKVPress**：使用分块选择而非全局 top-k
4. **ComposedPress**：不进行压缩，而是组合其他 Press
5. **DuoAttentionPress**：修改注意力计算方式，而非剪枝 KV

## 五、使用示例

### 使用 ScorerPress 的简单压缩
```python
from kvpress import KnormPress

# 只需指定压缩率，评分和剪枝逻辑已实现
press = KnormPress(compression_ratio=0.4)
with press(model):
    outputs = model.generate(...)
```

### 使用包装器实现复杂策略
```python
from kvpress import ChunkKVPress, SnapKVPress

# ChunkKVPress 包装 SnapKVPress，实现分块选择
press = ChunkKVPress(
    press=SnapKVPress(compression_ratio=0.4),
    chunk_length=20
)
with press(model):
    outputs = model.generate(...)
```

### 组合多个压缩方法
```python
from kvpress import ComposedPress, SnapKVPress, ThinKPress

# 先压缩序列长度，再压缩 key 维度
press = ComposedPress([
    SnapKVPress(compression_ratio=0.5),  # 压缩序列
    ThinKPress(key_channel_compression_ratio=0.5)  # 压缩维度
])
with press(model):
    outputs = model.generate(...)
```

## 六、总结

| 继承类别 | 适用场景 | 需要实现的方法 | 示例 |
|----------|----------|----------------|------|
| **ScorerPress** | 简单的"评分+剪枝"逻辑 | `score()` | KnormPress, SnapKVPress |
| **BasePress** | 复杂的压缩逻辑、包装器、维度压缩 | `compress()` | AdaKVPress, ThinKPress, ComposedPress |
| **其他 Press** | 扩展现有方法 | 根据需要重写 | PyramidKVPress |

**设计原则**：
- 如果你的压缩方法可以表达为"给每个 token 打分，然后保留分数最高的"，继承 **ScorerPress**
- 如果你的压缩逻辑更复杂（如包装其他方法、压缩维度、修改注意力机制等），继承 **BasePress**
