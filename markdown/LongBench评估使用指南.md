# LongBench 评估使用指南

## 核心机制

### evaluation.py 参数收集流程

```python
# 1. 配置加载优先级（从低到高）
默认值 (EvaluationConfig) 
  ↓
YAML 配置文件 (evaluate_config.yaml)
  ↓
命令行参数 (--dataset, --press_name 等)  # 最高优先级
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset` | str | "ruler" | 数据集名称：`longbench` 或 `longbench-e` |
| `data_dir` | str/None | None | 子任务名称（如 `hotpotqa`）或 `None`（全部任务） |
| `press_name` | str | "knorm" | 压缩方法名称 |
| `compression_ratio` | float | 1.0 | 压缩率（0.0-1.0） |
| `fraction` | float | 1.0 | 数据采样比例（0.0-1.0） |
| `model` | str | "meta-llama/..." | 模型名称 |
| `device` | str/None | None | 设备（None=自动检测） |

---

## 快速开始

```bash
# 进入评估目录
cd evaluation

# 最快验证（30秒）
python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5 --fraction 0.01

# 单任务完整评估（10分钟）
python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5

# 所有任务评估（2-4小时）
python evaluate.py --dataset longbench --press_name knorm --compression_ratio 0.5
```

---

## 参数配置方式

### 方式 1: 命令行参数（推荐快速测试）

```bash
python evaluate.py \
  --dataset longbench \
  --data_dir hotpotqa \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --press_name knorm \
  --compression_ratio 0.5 \
  --fraction 0.1 \
  --device cuda:0
```

### 方式 2: 修改配置文件

编辑 `evaluate_config.yaml`：
```yaml
dataset: "longbench"
data_dir: "hotpotqa"        # None = 所有任务
press_name: "knorm"
compression_ratio: 0.5
fraction: 1.0               # 1.0 = 全部数据
model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
device: null                # null = 自动检测
```

然后运行：
```bash
python evaluate.py
```

### 方式 3: 自定义配置文件

```bash
python evaluate.py --config_file my_config.yaml
```

---

## 核心参数详解

### 1. `--dataset` 参数

**作用**：指定数据集类型

```python
# evaluate.py 内部逻辑
DATASET_REGISTRY = {
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    # ... 其他数据集
}

# 加载数据集
df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test")
```

**选项**：
- `longbench`：普通版本，返回单个分数
- `longbench-e`：按长度分组，返回 `{"0-4k": x, "4-8k": y, "8k+": z}`

### 2. `--data_dir` 参数

**作用**：选择特定子任务

```python
# data_dir = None：加载所有子任务
df = load_dataset("Xnhyacinth/LongBench", split="test")  # 所有 21 个任务

# data_dir = "hotpotqa"：只加载 hotpotqa
df = load_dataset("Xnhyacinth/LongBench", "hotpotqa", split="test")  # 200 个样本
```

**可用子任务**（21个）：
```
# 多文档 QA
hotpotqa, 2wikimqa, musique, dureader

# 单文档 QA
narrativeqa, qasper, multifieldqa_en, multifieldqa_zh

# 摘要
gov_report, qmsum, multi_news, vcsum

# Few-shot
trec, triviaqa, samsum, lsht

# 合成任务
passage_count, passage_retrieval_en, passage_retrieval_zh

# 代码
lcc, repobench-p
```

### 3. `--fraction` 参数

**作用**：随机采样数据集

```python
# evaluate.py 实现
if fraction < 1.0:
    df = df.sample(frac=fraction, random_state=42)  # 固定种子，可重复
```

**示例**（hotpotqa 有 200 个样本）：
```bash
--fraction 1.0   # 200 个样本（全部）
--fraction 0.1   # 20 个样本（10%）
--fraction 0.01  # 2 个样本（1%）
```

**使用场景**：
| fraction | 样本数 | 时间 | 用途 |
|----------|--------|------|------|
| 0.01 | 2 | 30秒 | 验证代码能跑 |
| 0.1 | 20 | 5分钟 | 快速对比方法 |
| 1.0 | 200 | 10分钟 | 完整评估 |

### 4. `_e` 版本使用

**关键规则**：
```bash
# ✅ 正确：dataset 和 data_dir 都要改
python evaluate.py --dataset longbench-e --data_dir hotpotqa_e --press_name snapkv --compression_ratio 0.5

# ❌ 错误：只改了一个
python evaluate.py --dataset longbench --data_dir hotpotqa_e
python evaluate.py --dataset longbench-e --data_dir hotpotqa
```

**输出对比**：
```bash
# 普通版本
{"score": 65.8}

# _e 版本
{
  "0-4k": 78.5,
  "4-8k": 65.2,
  "8k+": 52.3
}
```

---

## 数据集结构（以 hotpotqa 为例）

```python
from datasets import load_dataset
ds = load_dataset('Xnhyacinth/LongBench', 'hotpotqa', split='test')

# 总样本数
len(ds)  # 200

# 每个样本的结构
sample = ds[0]
{
  "context": "Answer the question based on...Passage 1: ...",  # 长文本（~5-10k tokens）
  "question": "Question: Which case was brought first?",        # 问题
  "answers": ["Miller v. California"],                          # 标准答案列表
  "task": "hotpotqa",                                           # 任务名
  "length": 7845,                                               # 长度（用于_e分组）
  "max_new_tokens": 52                                          # 生成token数
}
```

**推理流程**：
1. 压缩 `context` 的 KV cache
2. 基于压缩后的 cache 生成答案
3. 比较生成答案与 `answers` 的匹配度

---

## 常用命令速查

### 基础测试

```bash
# 1. 超快验证（30秒）
python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5 --fraction 0.01

# 2. 单任务测试
python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5

# 3. 单任务 _e 版本
python evaluate.py --dataset longbench-e --data_dir hotpotqa_e --press_name knorm --compression_ratio 0.5

# 4. 所有任务
python evaluate.py --dataset longbench --press_name knorm --compression_ratio 0.5
```

### 对比不同压缩率

```bash
for ratio in 0.3 0.5 0.7; do
  python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio $ratio
done
```

### 对比不同方法

```bash
for press in knorm expected_attention snapkv streaming_llm random no_press; do
  python evaluate.py --dataset longbench --data_dir hotpotqa --press_name $press --compression_ratio 0.5
done
```

### 批量测试特定任务

```bash
# 只测试多文档 QA
for task in hotpotqa 2wikimqa musique; do
  python evaluate.py --dataset longbench --data_dir $task --press_name knorm --compression_ratio 0.5
done
```

### 所有 _e 版本任务

```bash
# 多文档 QA
python evaluate.py --dataset longbench-e --data_dir hotpotqa_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir 2wikimqa_e --press_name knorm --compression_ratio 0.5

# 单文档 QA
python evaluate.py --dataset longbench-e --data_dir narrativeqa_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir qasper_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir multifieldqa_en_e --press_name knorm --compression_ratio 0.5

# 摘要
python evaluate.py --dataset longbench-e --data_dir gov_report_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir multi_news_e --press_name knorm --compression_ratio 0.5

# Few-shot
python evaluate.py --dataset longbench-e --data_dir trec_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir triviaqa_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir samsum_e --press_name knorm --compression_ratio 0.5

# 合成任务
python evaluate.py --dataset longbench-e --data_dir passage_count_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir passage_retrieval_en_e --press_name knorm --compression_ratio 0.5

# 代码
python evaluate.py --dataset longbench-e --data_dir lcc_e --press_name knorm --compression_ratio 0.5
python evaluate.py --dataset longbench-e --data_dir repobench-p_e --press_name knorm --compression_ratio 0.5
```

---

## 结果分析

### 输出目录结构

```
results/
└── {dataset}__{data_dir}__{model}__{press}__{ratio}/
    ├── config.yaml          # 运行配置
    ├── predictions.csv      # 预测结果
    └── metrics.json         # 评估指标
```

### 查看结果

```bash
# 查看指标
cat results/longbench__hotpotqa__*/metrics.json

# 查看预测（前20行）
head -n 20 results/longbench__hotpotqa__*/predictions.csv

# 对比多个结果
cat results/longbench__hotpotqa__*__knorm__*/metrics.json
```

### Python 脚本对比

```python
import json
from pathlib import Path
import pandas as pd

results = {}
for p in Path('results').glob('longbench__hotpotqa__*__knorm__*/metrics.json'):
    parts = p.parent.name.split('__')
    ratio = parts[-1]
    with open(p) as f:
        results[ratio] = json.load(f)

df = pd.DataFrame(results).T
print(df.sort_index())
```

---

## 常见问题

### Q1: `--dataset longbench` 底层发生什么？

```python
# 步骤 1: 加载数据集
df = load_dataset("Xnhyacinth/LongBench", data_dir=None, split="test")
# 结果: 加载所有 21 个子任务

# 步骤 2: 格式化 prompt（每个任务有不同模板）
# 步骤 3: 批量推理
# 步骤 4: 计算所有任务的平均分数
```

### Q2: 如何只测试特定任务？

```bash
# 使用 --data_dir 参数
python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5
```

### Q3: 内存不足怎么办？

```bash
# 方案 1: 使用小模型
--model meta-llama/Meta-Llama-3.1-8B-Instruct

# 方案 2: FP8 量化
--fp8 true

# 方案 3: 限制上下文长度（只截断 context，不影响 question）
--max_context_length 4096

# 方案 4: 使用部分数据
--fraction 0.1
```

**重要**: `--max_context_length` 只截断 **context**（长文本），不会影响 **question**（用户问题）。详见 Q7。

### Q4: 如何指定 GPU？

```bash
# 方法 1: device 参数
python evaluate.py --device cuda:1 --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5

# 方法 2: 环境变量
CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset longbench --data_dir hotpotqa --press_name knorm --compression_ratio 0.5
```

### Q5: 查看可用的压缩方法

查看 `evaluate_registry.py` 的 `PRESS_REGISTRY`：
```python
PRESS_REGISTRY = {
    "knorm": KnormPress(),
    "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "random": RandomPress(),
    "no_press": None,
    # ... 更多方法
}
```

### Q6: 评估时间估算

使用 8B 模型，单个 A100 GPU：

| 配置 | 时间 | 说明 |
|------|------|------|
| 单任务, fraction=0.01 | 30秒 | 快速验证 |
| 单任务, fraction=0.1 | 5分钟 | 快速对比 |
| 单任务, fraction=1.0 | 10分钟 | 完整单任务 |
| 所有任务, fraction=0.1 | 30分钟 | 快速全局验证 |
| 所有任务, fraction=1.0 | 2-4小时 | 完整评估 |

### Q7: `--max_context_length` 会截断用户的问题吗？

**不会！** `--max_context_length` **只截断 context（长文本），不会影响 question（用户问题）**。

**代码实现**（pipeline.py）：
```python
# 步骤 1: 分别处理 context 和 question
context_ids = tokenizer.encode(context)        # 编码长文本
question_ids = tokenizer.encode(question)      # 编码问题

# 步骤 2: 只截断 context
if context_ids.shape[1] > max_context_length:
    context_ids = context_ids[:, :max_context_length]  # 截断 context

# 步骤 3: question 保持完整，不被截断
# question_ids 始终保持完整
```

**实际效果**（以 hotpotqa 为例）：
```python
# 原始数据
context = "Answer the question...Passage 1: ... (10000 tokens)"
question = "Question: Which case was brought first?"  # 约 10 tokens

# 设置 max_context_length=4096 后
context_truncated = context[:4096]  # 截断到 4096 tokens
question_unchanged = question        # 保持完整的 10 tokens

# 最终输入模型
total_tokens = 4096 + 10 = 4106 tokens
```

**为什么这样设计？**
- context 是**可压缩的背景信息**（长文档、参考资料）
- question 是**核心任务指令**（必须完整保留）
- 截断 context 可以节省内存，但不影响模型理解任务

**使用建议**：
```bash
# 场景 1: context 太长导致 OOM
--max_context_length 8192  # 截断 context 到 8k，question 仍然完整

# 场景 2: 测试不同上下文长度对性能的影响
--max_context_length 4096  # context 4k, question 完整
--max_context_length 8192  # context 8k, question 完整
--max_context_length 16384 # context 16k, question 完整
```

---



**最后更新**: 2026-02-04
