#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1
NB_GPU=2

DATA_ROOT='./data'
DATASET=ade
TASK=100-10
NAME=OURS
INCREMENTAL_METHOD=FT
STEPS_GLOBAL=5
TASK_NUM=6
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

CLASS_RATIO=0.7
SAMPLE_RATIO2=1.0
BATCH_SIZE=12
EPOCHS_LOCAL=12
ENTROPY_THRESHOLD=0.6
SOFT_PARAM=0.3
REGULAR_PARAM=0.1
MAX_PORTION=0.8
PORTION_STEP=0.05

SEED=2023
echo ${EPOCHS_GLOBAL}

OPTIONS="--pod local --pod_factor 0.001 --pod_logits --pseudo entropy --threshold 0.001 --classif_adaptive_factor --init_balanced --out_dis"

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/seed_${SEED}-ov/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} fl_main.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --use_entropy_detection --base_weights --dataset ${DATASET} --name ${NAME} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --class_ratio ${CLASS_RATIO} --sample_ratio2 ${SAMPLE_RATIO2} --batch_size ${BATCH_SIZE} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --seed ${SEED} --entropy_threshold ${ENTROPY_THRESHOLD} --max_portion ${MAX_PORTION} --portion_step ${PORTION_STEP} --soft_param ${SOFT_PARAM} --regular_param ${REGULAR_PARAM} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}" 

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
