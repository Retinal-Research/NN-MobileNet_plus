#!/bin/bash

# ========= Fixed Config =========
BATCH_SIZE=32
INPUT_SIZE=224
DATASET="odir"
WEIGHT_DECAY=1e-4
EPOCHS=300
MAIN_EVAL="auc"
NB_CLASSES=8
EXP_TYPE="down_odir_multi_swin"

# 你的图像根目录
DATA_PATH="/scratch/xinli38/data/MICCAI/image"

# 5折CSV所在目录（里面应有 train1.csv…train5.csv 和 test1.csv…test5.csv）
FOLDS_DIR="/scratch/xinli38/data/ODIR/5-folds"   

# ========= Swept Parameters =========
OPT_LIST=("adamp")
LR_LIST=("1e-3")
DROP_PATH="0.01"
MIXUP="0.0"
CUTMIX="0.0"

# ========= Loops: optimizer × lr × folds =========
for OPT in "${OPT_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    for FOLD in {3..4}; do

      TRAIN_CSV="${FOLDS_DIR}/train_fold${FOLD}.csv"
      TEST_CSV="${FOLDS_DIR}/test_fold${FOLD}.csv"

      if [[ ! -f "$TRAIN_CSV" ]] || [[ ! -f "$TEST_CSV" ]]; then
        echo "❌ Missing CSV for fold ${FOLD}: $TRAIN_CSV or $TEST_CSV"
        exit 1
      fi

      EXP_NAME="${DATASET}_lr${LR}_drop${DROP_PATH}_mix${MIXUP}_cut${CUTMIX}_opt${OPT}_bz${BATCH_SIZE}_fold${FOLD}"
      OUTPUT_DIR="Experiment/1_0/${EXP_TYPE}/${EXP_NAME}"
      LOG_DIR="$OUTPUT_DIR"
      mkdir -p "$OUTPUT_DIR"

      echo "🚀 Starting: OPT=${OPT}, LR=${LR}, FOLD=${FOLD}"
      python main.py \
        --data_path "$DATA_PATH" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --input_size "$INPUT_SIZE" \
        --data_set "$DATASET" \
        --drop_path "$DROP_PATH" \
        --weight_decay "$WEIGHT_DECAY" \
        --epochs "$EPOCHS" \
        --main_eval "$MAIN_EVAL" \
        --opt "$OPT" \
        --nb_classes "$NB_CLASSES" \
        --mixup "$MIXUP" \
        --cutmix "$CUTMIX" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --fold_train "$TRAIN_CSV" \
        --fold_test "$TEST_CSV" \
        --smoothing 0.0

    done
  done
done