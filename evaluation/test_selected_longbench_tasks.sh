#!/bin/bash

# LongBench 选择性任务测试脚本
# 用法: bash test_selected_longbench_tasks.sh

echo "=== LongBench 选择性任务测试 ==="

# 配置
PRESS="knorm"
RATIO=0.5
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# 只测试 QA 相关任务
TASKS=("hotpotqa" "2wikimqa" "musique" "multifieldqa_en")

echo ""
echo "将测试以下任务:"
for task in "${TASKS[@]}"; do
  echo "  - $task"
done
echo ""
echo "压缩方法: $PRESS"
echo "压缩率: $RATIO"
echo "模型: $MODEL"
echo ""

# 询问是否继续
read -p "是否继续? (Y/N): " confirm
if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
  echo "已取消"
  exit 0
fi

# 执行测试
success_count=0
fail_count=0

for task in "${TASKS[@]}"; do
  echo ""
  echo "========================================"
  echo "正在测试任务: $task"
  echo "========================================"
  echo ""
  
  cmd="python evaluate.py --dataset longbench --data_dir $task --model $MODEL --press_name $PRESS --compression_ratio $RATIO"
  echo "执行命令: $cmd"
  echo ""
  
  $cmd
  
  if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 任务 $task 完成"
    ((success_count++))
  else
    echo ""
    echo "✗ 任务 $task 失败"
    ((fail_count++))
  fi
done

# 总结
echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"
echo "成功: $success_count / ${#TASKS[@]}"
echo "失败: $fail_count / ${#TASKS[@]}"
echo ""
echo "结果保存在 ./results 目录中"
