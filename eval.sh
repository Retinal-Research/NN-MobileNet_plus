#!/bin/bash
# === Fixed Config ===
BATCH_SIZE=1
INPUT_SIZE=224
DATASET="MICCAI"
WEIGHT_DECAY=1e-4
EPOCHS=120
MAIN_EVAL="auc"
OPT="adamp"
NB_CLASSES=5
EXP_TYPE=test

TRAIN_CSV="/scratch/xinli38/data/MICCAI/label/MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
TEST_CSV="/scratch/xinli38/data/MICCAI/label/MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
DATA_PATH="/scratch/xinli38/data/MICCAI/image"

# === Swept Parameters ===
LR_LIST=("1e-4" "2e-4" "1e-3")

# === Loop ===

LR=1e-4


# Construct experiment name
EXP_NAME="${DATASET}_lr${LR}_drop${DROP_PATH}_mix${MIXUP}_cut${CUTMIX}_opt${OPT}"
OUTPUT_DIR="/scratch/xinli38/nn-mobilenet++/Experiment/$EXP_TYPE/$EXP_NAME"
LOG_DIR="$OUTPUT_DIR"

python main.py \
--data_path $DATA_PATH \
--batch_size $BATCH_SIZE \
--lr $LR \
--input_size $INPUT_SIZE \
--data_set $DATASET \
--weight_decay $WEIGHT_DECAY \
--epochs $EPOCHS \
--main_eval $MAIN_EVAL \
--opt $OPT \
--nb_classes $NB_CLASSES \
--output_dir $OUTPUT_DIR \
--log_dir $LOG_DIR \
--fold_train $TRAIN_CSV \
--fold_test $TEST_CSV \
--eval True
