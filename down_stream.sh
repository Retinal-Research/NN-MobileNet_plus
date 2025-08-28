#!/bin/bash

# === Fixed Config ===
BATCH_SIZE=32
INPUT_SIZE=224
DATASET="idrid"
WEIGHT_DECAY=5e-4
EPOCHS=1200
MAIN_EVAL="auc"
NB_CLASSES=5
EXP_TYPE="down_odir_bin"

TRAIN_CSV="/scratch/xinli38/data/IDRID/train.csv"
TEST_CSV="/scratch/xinli38/data/IDRID/test.csv"
DATA_PATH="/scratch/xinli38/data/MICCAI/image"

# === Swept Parameters ===
OPT_LIST=("adamp" "adamw" "radam")
LR_LIST=("2e-4" "3e-4")
DROP_PATH="0.15"
MIXUP="0.0"
CUTMIX="0.0"

# === Nested Loop ===
for OPT in "${OPT_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do

    # Construct experiment name
    EXP_NAME="${DATASET}_lr${LR}_drop${DROP_PATH}_mix${MIXUP}_cut${CUTMIX}_opt${OPT}_bz${BATCH_SIZE}_cut&mix${MIXUP}_${CUTMIX}"
    OUTPUT_DIR="/scratch/xinli38/nn-mobilenet++/Experiment/1_0/$EXP_TYPE/$EXP_NAME"
    LOG_DIR="$OUTPUT_DIR"

    echo "🚀 Starting: OPT=$OPT, LR=$LR"

    python main.py \
      --data_path $DATA_PATH \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --input_size $INPUT_SIZE \
      --data_set $DATASET \
      --drop_path $DROP_PATH \
      --weight_decay $WEIGHT_DECAY \
      --epochs $EPOCHS \
      --main_eval $MAIN_EVAL \
      --opt $OPT \
      --nb_classes $NB_CLASSES \
      --mixup $MIXUP \
      --cutmix $CUTMIX \
      --output_dir $OUTPUT_DIR \
      --log_dir $LOG_DIR \
      --fold_train $TRAIN_CSV \
      --fold_test $TEST_CSV \
      --smoothing 0.05
  done
done