#!/bin/bash

# === Fixed Config ===
BATCH_SIZE=4
INPUT_SIZE=224
DATASET="MICCAI"
WEIGHT_DECAY=1e-4
EPOCHS=120
MAIN_EVAL="auc"
OPT="adamp"
NB_CLASSES=5
EXP_TYPE=dsc_x

TRAIN_CSV="/scratch/xinli38/data/MICCAI/label/MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
TEST_CSV="/scratch/xinli38/data/MICCAI/label/MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
DATA_PATH="/scratch/xinli38/data/MICCAI/image"

# === Swept Parameters ===
LR_LIST=("1e-3" "1e-3" "1e-3")
DROP_PATH_LIST=("0.1" "0.1" "0.1")


MIXUP_LIST=("0.0" "0.2" "0.4")
CUTMIX_LIST=("1.0" "0.8" "0.6")

# === Loop ===
for i in {0..2}; do
  LR=${LR_LIST[$i]}
  DROP_PATH=${DROP_PATH_LIST[$i]}
  MIXUP=${MIXUP_LIST[$i]}
  CUTMIX=${CUTMIX_LIST[$i]}

  # Construct experiment name
  EXP_NAME="${DATASET}_lr${LR}_drop${DROP_PATH}_mix${MIXUP}_cut${CUTMIX}_opt${OPT}"
  OUTPUT_DIR="/scratch/xinli38/nn-mobilenet++/Experiment/1_0/$EXP_TYPE/$EXP_NAME"
  LOG_DIR="$OUTPUT_DIR"

  echo "🚀 Starting run $((i+1))/3: LR=$LR, DropPath=$DROP_PATH, Mixup=$MIXUP, CutMix=$CUTMIX"

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
    --fold_test $TEST_CSV
done