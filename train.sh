#!/bin/bash

# ================= 配置区 =================
# 1. 指定使用哪一张显卡
GPU_ID=6

# 2. 配置文件路径
CONFIG_PATH="./configs/config_kitti.yaml"

# 3. 接收实验 Extra Tag (第一个参数)
# 如果没提供参数，默认叫 "base"
EXTRA_TAG=${1:-"base"}

# 4. 提取配置文件名作为前缀
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml | sed 's/config_//')

# 5. 自动获取当前时间 (月日_时分)
TIME_STAMP=$(date +%m%d_%H%M)

# 6. 实验日志目录名 (加入 EXTRA_TAG)
# 例如：Exp_kitti_coord_decouple_0309_1530
EXP_NAME="Exp_${CONFIG_NAME}_${EXTRA_TAG}_${TIME_STAMP}"

# 7. 日志保存目录
LOG_DIR="nhlogs"

# 8. 控制台输出的日志文件名 (同样加入 EXTRA_TAG)
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${EXTRA_TAG}_${TIME_STAMP}.log"
# ==========================================

# 确保日志文件夹存在
mkdir -p $LOG_DIR

# 检查是否需要后台运行 (第二个参数)
USE_NOHUP=false
if [[ "$2" == "nohup" ]]; then
    USE_NOHUP=true
fi

echo "--- 启动配置 ---"
echo "使用显卡  : $GPU_ID"
echo "配置来源  : $CONFIG_NAME"
echo "实验标签  : $EXTRA_TAG"
echo "实验名称  : $EXP_NAME"
echo "日志文件  : $LOG_FILE"
echo "运行模式  : $( ${USE_NOHUP} && echo "后台(nohup)" || echo "前台" )"
echo "----------------"

# 执行命令
if [ "$USE_NOHUP" = true ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python main.py \
        --logdir=$EXP_NAME \
        --config=$CONFIG_PATH > $LOG_FILE 2>&1 &
    
    echo "任务已提交后台，PID: $!"
    echo "查看日志: tail -f $LOG_FILE"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --logdir=$EXP_NAME \
        --config=$CONFIG_PATH 2>&1 | tee $LOG_FILE
fi