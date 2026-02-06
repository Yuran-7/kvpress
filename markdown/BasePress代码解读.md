# BasePress 代码解读：KV Cache 压缩的执行流程

本文档分为两部分：
- **第一部分**：钩子机制与零入侵设计（核心概念）
- **第二部分**：KV Cache 压缩的执行流程（具体实现）

---

# 第一部分：钩子机制与零入侵设计

## 什么是钩子（Hooks）？

### 概念解释

**钩子（Hook）** 是编程中非常常见的设计模式，它允许你在某个函数或模块执行的特定时刻**插入自定义代码**，而无需修改原始代码本身。

可以把钩子想象成一个"挂钩点"：
- 原始代码在执行到特定位置时，会**主动检查**是否有钩子函数注册
- 如果有，就**自动调用**这些钩子函数
- 钩子函数执行完毕后，原始代码继续运行

### 钩子的生活类比

想象你在一家咖啡店：
```
正常流程：
1. 点单 → 2. 制作咖啡 → 3. 交付咖啡

添加钩子后：
1. 点单
2. [Hook: 记录订单日志]      ← 插入的钩子
3. 制作咖啡
4. [Hook: 检查质量]          ← 插入的钩子
5. 交付咖啡
6. [Hook: 发送满意度调查]    ← 插入的钩子
```

咖啡店的核心流程不变，但通过钩子，你可以在关键时刻插入额外的操作。

### PyTorch 中的钩子

在 PyTorch 中，钩子主要用于**监控和修改神经网络层的输入输出**。

#### `register_forward_hook` 是什么？

`register_forward_hook` 是 PyTorch 的 `nn.Module` 提供的一个方法，它允许你在**任何层的前向传播完成后**插入自定义函数。

**核心特点：**
- **通用性**：适用于任何 `nn.Module` 子类（Linear、Conv2d、Transformer、自定义层等）
- **调用时机**：在层的 `forward()` 方法**执行完毕后**自动调用
- **参数**：钩子函数接收三个参数：`(module, input, output)`
  - `module`：触发钩子的层对象
  - `input`：传入该层的输入（tuple 形式）
  - `output`：该层的输出
- **返回值**：可以返回修改后的 output，也可以返回 None（不修改）

#### 完整示例：简单的 CNN 分类器

让我们用一个简单的图像分类网络来演示钩子的通用性：

```python
import torch
import torch.nn as nn

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 卷积层2
        self.fc1 = nn.Linear(32 * 8 * 8, 128)                    # 全连接层1
        self.fc2 = nn.Linear(128, 10)                            # 全连接层2
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 3, 32, 32] -> [batch, 16, 16, 16]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 16, 16, 16] -> [batch, 32, 8, 8]
        x = x.view(x.size(0), -1)                # 展平
        x = self.relu(self.fc1(x))               # [batch, 2048] -> [batch, 128]
        x = self.fc2(x)                          # [batch, 128] -> [batch, 10]
        return x

# 创建模型
model = SimpleCNN()

# 定义钩子函数
def monitor_hook(module, input, output):
    """
    监控钩子：打印层的信息
    
    参数：
        module: 触发钩子的层（如 Conv2d, Linear 等）
        input: 输入到该层的数据（tuple 形式）
        output: 该层的输出
    """
    print(f"[Hook] Layer: {module.__class__.__name__}")
    print(f"        Input shape: {input[0].shape}")
    print(f"        Output shape: {output.shape}")
    print("-" * 50)
    return None  # 不修改输出

def modify_hook(module, input, output):
    """
    修改钩子：对输出进行修改
    """
    # 例如：将输出乘以 0.5（类似于 dropout 的效果）
    return output * 0.5

# ============================================================
# 示例1：监控所有卷积层
# ============================================================
print("=" * 60)
print("示例1：为卷积层注册监控钩子")
print("=" * 60)

# 为 conv1 和 conv2 注册钩子
handle1 = model.conv1.register_forward_hook(monitor_hook)
handle2 = model.conv2.register_forward_hook(monitor_hook)

# 创建测试输入
x = torch.randn(2, 3, 32, 32)  # batch_size=2, 3通道, 32x32图像

# 前向传播（钩子会自动触发）
output = model(x)
print(f"Final output shape: {output.shape}\n")

# 移除钩子
handle1.remove()
handle2.remove()

# ============================================================
# 示例2：为全连接层注册修改钩子
# ============================================================
print("=" * 60)
print("示例2：为全连接层注册修改钩子")
print("=" * 60)

# 不使用钩子的输出
output_without_hook = model(x)
print(f"Without hook: {output_without_hook[0, :5]}")  # 打印第一个样本的前5个值

# 注册修改钩子到 fc1
handle = model.fc1.register_forward_hook(modify_hook)

# 使用钩子后的输出
output_with_hook = model(x)
print(f"With hook:    {output_with_hook[0, :5]}")     # 输出会发生变化

handle.remove()

# ============================================================
# 示例3：为所有层注册钩子
# ============================================================
print("\n" + "=" * 60)
print("示例3：为模型的所有层注册钩子")
print("=" * 60)

handles = []
# 遍历模型的所有子模块
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):  # 只为卷积层和全连接层注册
        handle = module.register_forward_hook(monitor_hook)
        handles.append(handle)

# 前向传播
output = model(x)

# 批量移除所有钩子
for handle in handles:
    handle.remove()
```

**输出示例：**
```
============================================================
示例1：为卷积层注册监控钩子
============================================================
[Hook] Layer: Conv2d
        Input shape: torch.Size([2, 3, 32, 32])
        Output shape: torch.Size([2, 16, 32, 32])
--------------------------------------------------
[Hook] Layer: Conv2d
        Input shape: torch.Size([2, 16, 16, 16])
        Output shape: torch.Size([2, 32, 16, 16])
--------------------------------------------------
Final output shape: torch.Size([2, 10])

============================================================
示例2：为全连接层注册修改钩子
============================================================
Without hook: tensor([-0.2341,  0.1234, -0.5678,  0.9012, -0.3456], grad_fn=<...>)
With hook:    tensor([-0.1543,  0.0812, -0.3741,  0.5934, -0.2276], grad_fn=<...>)
```

#### 钩子的调用时机详解

```python
class MyLayer(nn.Module):
    def forward(self, x):
        print("  → forward() 开始执行")
        output = x * 2
        print("  → forward() 执行完毕，准备返回")
        return output  # ← 在这里返回后，forward hook 被触发

layer = MyLayer()

def my_hook(module, input, output):
    print("    → Hook 被调用！")
    print(f"      Input: {input[0].item()}, Output: {output.item()}")
    return output

layer.register_forward_hook(my_hook)

x = torch.tensor(5.0)
print("调用 layer(x):")
result = layer(x)
print(f"最终结果: {result.item()}\n")

# 输出：
# 调用 layer(x):
#   → forward() 开始执行
#   → forward() 执行完毕，准备返回
#     → Hook 被调用！
#       Input: 5.0, Output: 10.0
# 最终结果: 10.0
```

**时序图：**
```
用户调用 module(x)
    ↓
module.forward(x) 开始执行
    ↓
计算逻辑（层的实际计算）
    ↓
module.forward(x) 返回 output
    ↓
【触发点】自动调用 forward_hook(module, input, output)
    ↓
    ├─ 如果 hook 返回新值 → 使用新值作为最终输出
    └─ 如果 hook 返回 None → 使用原始 output
    ↓
返回结果给用户
```

#### 关键要点

1. **自动触发**：不需要手动调用，层的 `forward()` 执行完后自动触发
2. **通用性强**：任何 `nn.Module` 都可以注册钩子（Conv2d、Linear、LSTM、Transformer、自定义层等）
3. **可观察性**：可以查看任何层的中间输出，用于调试和分析
4. **可修改性**：可以修改层的输出，实现动态行为调整
5. **可移除性**：通过 `handle.remove()` 可以随时移除，不影响原始功能
6. **无侵入性**：不需要修改层的源代码

### 钩子的常见应用场景

1. **特征提取**：获取中间层的输出用于可视化或分析
2. **梯度修改**：在反向传播时修改梯度（gradient hooks）
3. **调试和监控**：打印中间结果、统计信息
4. **模型修改**：在不改变原始代码的情况下修改模型行为（如本项目的 KV cache 压缩）

## BasePress 的 `__call__` 函数

### 函数作用

`__call__` 方法将 `BasePress` 实例变成一个**上下文管理器（Context Manager）**，让你可以使用 `with` 语句：

```python
@contextmanager
def __call__(self, model: PreTrainedModel) -> Generator:
    """将压缩方法应用到模型的上下文管理器"""
    
    # 1. 初始化阶段
    self.post_init_from_model(model)
    hooks = []
    
    try:
        # 2. 注册钩子阶段
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for layer in language_model.layers:
            if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                continue
            layer.self_attn.rotary_emb = language_model.rotary_emb
            # 核心：为每个 attention 层注册 forward hook
            hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
        
        # 3. yield：让出控制权，让用户执行模型推理
        yield
        
    finally:
        # 4. 清理阶段：移除所有钩子
        for forward_hook in hooks:
            forward_hook.remove()
```

### 执行流程

```python
# 创建压缩器实例
press = KnormPress(compression_ratio=0.5)

# 使用 with 语句
with press(model):  # ← 调用 __call__ 方法
    # 进入 with 块前：
    # 1. 执行 __call__ 的 try 块
    # 2. 为所有层注册 forward_hook
    # 3. 执行到 yield，暂停
    
    outputs = model(input_ids, past_key_values=cache)
    # 在这里，模型每层执行时都会自动触发 forward_hook
    
# 退出 with 块后：
# 1. 执行 finally 块
# 2. 移除所有 hooks
# 3. 模型恢复原状
```

### 为什么使用上下文管理器？

使用 `with` 语句的好处：

1. **自动清理**：无论是否发生异常，钩子都会被正确移除
2. **作用域明确**：压缩只在 `with` 块内生效
3. **资源管理**：防止内存泄漏（未移除的钩子会一直存在）
4. **代码优雅**：用户使用时非常简洁

```python
# 不好的方式（手动管理）
press = KnormPress(compression_ratio=0.5)
hooks = press.register_hooks(model)
try:
    outputs = model(input_ids)
finally:
    press.remove_hooks(hooks)  # 容易忘记

# 好的方式（自动管理）
with press(model):
    outputs = model(input_ids)  # 简洁且安全
```

## `with press(model)` 的作用

### 完整语义

```python
with press(model):
    outputs = model(input_ids, past_key_values=cache)
```

等价于：

```python
# 1. 调用 press.__call__(model)
context_manager = press(model)

# 2. 进入上下文（执行到 yield 前的代码）
#    - 为每层注册 forward_hook
context_manager.__enter__()

try:
    # 3. 执行用户代码
    outputs = model(input_ids, past_key_values=cache)
    #    每层执行时都会触发 forward_hook
    #    在 hook 中执行 KV cache 压缩
finally:
    # 4. 退出上下文（执行 finally 块）
    #    - 移除所有 forward_hook
    context_manager.__exit__(None, None, None)
```

### 钩子的生命周期

```
时间线：
─────────────────────────────────────────────────────────────
1. with press(model):          ← 进入上下文
   ├─ 注册 Layer 0 的 hook
   ├─ 注册 Layer 1 的 hook
   ├─ ...
   └─ 注册 Layer N 的 hook
   
2. outputs = model(input_ids)  ← 执行推理
   ├─ Embedding 层执行
   ├─ Layer 0 执行 → 触发 Layer 0 的 hook → 压缩 KV
   ├─ Layer 1 执行 → 触发 Layer 1 的 hook → 压缩 KV
   ├─ ...
   └─ Layer N 执行 → 触发 Layer N 的 hook → 压缩 KV
   
3. 退出 with 块               ← 退出上下文
   ├─ 移除 Layer 0 的 hook
   ├─ 移除 Layer 1 的 hook
   ├─ ...
   └─ 移除 Layer N 的 hook

4. 再次调用 model(input_ids)  ← 钩子已移除，正常执行
   └─ 不会触发任何压缩
```

## 钩子与零入侵（Zero Intrusion）

### 什么是零入侵？

**零入侵**意味着：
- 不修改 transformers 库的源代码
- 不需要继承或替换原始类
- 不需要重新实现模型的前向传播
- 只是在外部"挂钩"到模型的执行流程

### BasePress：零入侵设计

```python
# ✅ BasePress 的方式（零入侵）
from transformers import AutoModelForCausalLM
from kvpress import KnormPress

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
press = KnormPress(compression_ratio=0.5)

# 原始模型完全没有被修改
with press(model):
    # 只在这个 with 块内启用压缩
    outputs = model(input_ids, past_key_values=cache)

# 退出后，模型恢复原状
outputs = model(input_ids)  # 正常运行，无压缩
```

**零入侵的实现原理：**
1. 使用 PyTorch 的 `register_forward_hook` API
2. 钩子只是"监听"层的执行，不修改层的代码
3. 钩子可以随时添加和移除
4. transformers 库的代码保持原样

### Patch 方式：有入侵性

相比之下，"monkey patching" 是**有入侵性**的：

```python
# ❌ Patch 的方式（有入侵）
import transformers.models.llama.modeling_llama as llama_module

# 保存原始方法
original_forward = llama_module.LlamaAttention.forward

# 替换方法
def patched_forward(self, *args, **kwargs):
    # 修改后的前向传播逻辑
    output = original_forward(self, *args, **kwargs)
    # 压缩 KV cache
    return modified_output

# 全局替换（影响所有 LlamaAttention 实例）
llama_module.LlamaAttention.forward = patched_forward

# 所有使用 Llama 模型的代码都会受到影响！
model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# model1 和 model2 都被 patch 了，无法恢复
```

**Patch 的问题：**
1. **全局影响**：修改会影响所有实例
2. **难以恢复**：需要手动保存和恢复原始方法
3. **冲突风险**：多个库同时 patch 会冲突
4. **调试困难**：修改了源代码行为，难以追踪问题

### 对比总结

| 特性 | BasePress (钩子) | Patch |
|------|-----------------|-------|
| **入侵性** | ❌ 零入侵 | ✅ 有入侵 |
| **作用范围** | 特定模型实例 | 全局所有实例 |
| **可恢复性** | 自动恢复 | 需要手动恢复 |
| **冲突风险** | 低 | 高 |
| **代码清晰度** | 高（用 with） | 低（隐式修改） |
| **调试难度** | 容易 | 困难 |

### 实际对比示例

```python
# BasePress：精确控制
model1 = AutoModelForCausalLM.from_pretrained("model1")
model2 = AutoModelForCausalLM.from_pretrained("model2")

press1 = KnormPress(compression_ratio=0.5)
press2 = SnapKVPress(window_size=32)

# 只对 model1 应用 KnormPress
with press1(model1):
    output1 = model1(input_ids)

# 只对 model2 应用 SnapKVPress
with press2(model2):
    output2 = model2(input_ids)

# 两个模型独立工作，互不影响
output3 = model1(input_ids)  # 无压缩
output4 = model2(input_ids)  # 无压缩
```

```python
# Patch：全局影响，难以控制
patch_llama_attention()  # 全局 patch

model1 = AutoModelForCausalLM.from_pretrained("model1")
model2 = AutoModelForCausalLM.from_pretrained("model2")

# 两个模型都被 patch 了！无法分别控制
output1 = model1(input_ids)  # 被 patch
output2 = model2(input_ids)  # 被 patch

# 要恢复需要手动 unpatch
unpatch_llama_attention()
```

## 钩子机制的优势总结

1. **非侵入性**：不修改原始代码，保持库的完整性
2. **灵活性**：可以动态添加和移除
3. **可组合性**：多个钩子可以共存
4. **局部性**：只影响特定的模型实例
5. **安全性**：出错时不会破坏原始功能
6. **可维护性**：代码结构清晰，易于理解和维护

---

# 第二部分：KV Cache 压缩的执行流程

## 问题回答

### 第0层的执行流程

**问：在进入第0层之前是不是已经有了 hidden_states？**

**答：是的。** 在进入第0层（layer 0）之前，hidden_states 已经存在。这些 hidden_states 来自模型的 **embedding 层**，它将输入的 token IDs 转换为向量表示。

具体流程：
```
Input Token IDs → Embedding Layer → hidden_states (初始) → Layer 0
```

**问：在第0层中 compress 函数传入的 keys 和 values 是在第0层中由隐藏状态生成的吗？**

**答：是的。** 在第0层的 attention 模块中：

1. **hidden_states 输入**：来自 embedding 层的初始向量
2. **生成 Q, K, V**：第0层的 self_attn 模块使用这些 hidden_states 通过线性投影（q_proj, k_proj, v_proj）生成 Query、Key、Value
3. **执行 attention**：计算注意力权重并更新 KV cache
4. **触发 forward_hook**：在第0层 attention 计算完成后，forward_hook 被调用
5. **提取并压缩**：从 cache 中提取第0层刚刚生成的 keys 和 values，传入 compress 函数进行压缩

### 第1层的执行流程

**第1层的 hidden_states 来自哪里？**

第1层的 hidden_states 来自**第0层的输出**。每一层的输出会作为下一层的输入。

具体流程：
```
Layer 0 输入: hidden_states (from embedding)
         ↓
Layer 0 Attention + FFN
         ↓
Layer 0 输出: hidden_states (更新后)
         ↓
Layer 1 输入: hidden_states (from Layer 0)
         ↓
Layer 1 Attention: 生成新的 K, V
         ↓
Layer 1 forward_hook: 压缩 Layer 1 的 K, V
```

**第1层的 keys 和 values 是如何生成的？**

与第0层相同，第1层的 keys 和 values 是在**第1层的 self_attn 模块中**，使用**第1层的输入 hidden_states**（即第0层的输出）通过线性投影生成的。

## 完整执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                      模型前向传播流程                              │
└─────────────────────────────────────────────────────────────────┘

Input Token IDs: [token_1, token_2, ..., token_n]
       ↓
┌──────────────────┐
│  Embedding Layer │  将 token IDs 转换为向量
└──────────────────┘
       ↓
hidden_states_0 (初始隐藏状态)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Layer 0 (第0层)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 输入: hidden_states_0                                        │
│     ↓                                                            │
│  2. Self-Attention 模块:                                         │
│     • Q_0 = q_proj(hidden_states_0)                             │
│     • K_0 = k_proj(hidden_states_0)  ←── 生成第0层的 keys       │
│     • V_0 = v_proj(hidden_states_0)  ←── 生成第0层的 values     │
│     • attention_output = Attention(Q_0, K_0, V_0)               │
│     • 将 K_0, V_0 存入 cache.layers[0]                          │
│     ↓                                                            │
│  3. forward_hook 被触发:                                         │
│     • 从 kwargs 获取 hidden_states_0                            │
│     • 从 cache.layers[0] 提取 keys, values (即 K_0, V_0)        │
│     • 调用 compress(module, hidden_states_0, K_0, V_0, ...)     │
│     • 将压缩后的 keys, values 写回 cache.layers[0]              │
│     ↓                                                            │
│  4. Feed-Forward Network (FFN)                                  │
│     ↓                                                            │
│  5. 输出: hidden_states_1                                        │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Layer 1 (第1层)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 输入: hidden_states_1 (来自 Layer 0 的输出)                  │
│     ↓                                                            │
│  2. Self-Attention 模块:                                         │
│     • Q_1 = q_proj(hidden_states_1)                             │
│     • K_1 = k_proj(hidden_states_1)  ←── 生成第1层的 keys       │
│     • V_1 = v_proj(hidden_states_1)  ←── 生成第1层的 values     │
│     • attention_output = Attention(Q_1, K_1, V_1)               │
│     • 将 K_1, V_1 存入 cache.layers[1]                          │
│     ↓                                                            │
│  3. forward_hook 被触发:                                         │
│     • 从 kwargs 获取 hidden_states_1                            │
│     • 从 cache.layers[1] 提取 keys, values (即 K_1, V_1)        │
│     • 调用 compress(module, hidden_states_1, K_1, V_1, ...)     │
│     • 将压缩后的 keys, values 写回 cache.layers[1]              │
│     ↓                                                            │
│  4. Feed-Forward Network (FFN)                                  │
│     ↓                                                            │
│  5. 输出: hidden_states_2                                        │
└─────────────────────────────────────────────────────────────────┘
       ↓
     ... (后续层类似)
```

## forward_hook 详细解析

让我们深入分析 `forward_hook` 函数的执行时机和作用：

```python
def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
    """
    在 attention 层的前向传播 **完成后** 自动调用的钩子函数
    """
    # 1. 提取本层的输入 hidden_states
    hidden_states = kwargs["hidden_states"]
    
    # 2. 获取 KV cache 对象
    cache = kwargs["past_key_values"]
    cache_layer = cache.layers[module.layer_idx]
    q_len = hidden_states.shape[1]

    # 3. 判断是否还在预填充阶段（只在预填充时压缩）
    if kwargs["cache_position"][-1] > q_len:
        return output  # 已经进入生成阶段，不压缩

    # 4. 从 cache 中提取本层刚刚生成的 keys 和 values
    #    这些 keys 和 values 是在本层 attention 计算时生成并存入 cache 的
    keys, values = extract_keys_and_values(cache, module.layer_idx)

    # 5. 调用具体的压缩方法
    #    传入本层的输入 hidden_states 和本层生成的 keys, values
    keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

    # 6. 将压缩后的 keys 和 values 写回 cache
    if isinstance(cache, QuantizedCache):
        # 处理量化 cache
        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.cumulative_length = keys.shape[2]
    else:
        # 处理普通 cache
        cache_layer.keys = keys
        cache_layer.values = values

    return output
```

### 关键时机理解

**重要：** `forward_hook` 是在 attention 层的前向传播**完成后**才被调用的，因此：

1. **keys 和 values 已经生成**：在 hook 被调用时，当前层的 attention 计算已经完成，keys 和 values 已经通过 k_proj 和 v_proj 生成并存入 cache
2. **提取的是本层生成的 KV**：`extract_keys_and_values(cache, module.layer_idx)` 提取的是**当前层刚刚生成**的 keys 和 values
3. **hidden_states 是本层的输入**：传入 compress 函数的 hidden_states 是**本层的输入**（对第0层是 embedding 输出，对第1层是第0层的输出）



## 代码执行顺序总结

1. **注册 hooks**：进入 `with press(model)` 上下文时，为每层的 self_attn 注册 forward_hook
2. **执行前向传播**：调用 `model(input_ids, past_key_values=cache)`
3. **逐层处理**：
   - Embedding → hidden_states_0
   - Layer 0 attention 计算 → 生成 K_0, V_0 → 触发 hook → 压缩 K_0, V_0
   - Layer 0 FFN → hidden_states_1
   - Layer 1 attention 计算 → 生成 K_1, V_1 → 触发 hook → 压缩 K_1, V_1
   - ...依次类推
4. **移除 hooks**：退出 `with` 上下文时，移除所有 hooks

## 结论

- **第0层**：输入来自 embedding 层，keys/values 由第0层的 attention 使用这些输入生成
- **第1层**：输入来自第0层的输出，keys/values 由第1层的 attention 使用第1层的输入生成
- **每层都独立**：每层都用自己的输入生成自己的 KV，然后在该层的 forward_hook 中进行压缩
- **压缩时机**：在每层 attention 计算完成后，通过 forward_hook 立即对该层的 KV 进行压缩

这种设计使得每层的 KV cache 可以根据该层的特点独立压缩，提供了很大的灵活性。
