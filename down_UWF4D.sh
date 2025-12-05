#!/bin/bash

# ======================== #
#    Parameter Settings    #
# ======================== #

WIDTH_MULT=1
NB_CLASSES=2
BATCH_SIZE=32
INPUT_SIZE=224
DATASET="UWF4DR"
DROP_PATH=0.01
WEIGHT_DECAY=5e-4
EPOCHS=200
MAIN_EVAL="auc"
FOLD_TRAIN="/scratch/xinli38/data/UWF4DR/task1/train.csv"
FOLD_TEST="/scratch/xinli38/data/UWF4DR/task1/test.csv"
BASE_DIR="/scratch/xinli38/challenge/Experiments/test/"

# Define optimizers and learning rates
OPT_LIST=("adamw")
LR_LIST=(1e-4)

# ======================== #
#     Training Loops       #
# ======================== #

# Outer loop over optimizers
for OPT in "${OPT_LIST[@]}"
do
  # Inner loop over learning rates
  for LR in "${LR_LIST[@]}"
  do
    echo "🚀 Starting training with OPT=${OPT}, LR=${LR}"

    # Construct output and log directory
    OUTPUT_DIR="${BASE_DIR}/${OPT}_lr_${LR}_dp_${DROP_PATH}"
    LOG_DIR="${BASE_DIR}/${OPT}_lr_${LR}_dp_${DROP_PATH}"

    # Create directories if they do not exist
    mkdir -p $OUTPUT_DIR
    mkdir -p $LOG_DIR

    # Launch training
    python main.py \
      --nb_classes $NB_CLASSES \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --input_size $INPUT_SIZE \
      --data_set $DATASET \
      --drop_path $DROP_PATH \
      --weight_decay $WEIGHT_DECAY \
      --epochs $EPOCHS \
      --main_eval $MAIN_EVAL \
      --opt $OPT \
      --fold_train $FOLD_TRAIN \
      --fold_test $FOLD_TEST \
      --output_dir $OUTPUT_DIR \
      --log_dir $LOG_DIR \
      --eval True
  done
done