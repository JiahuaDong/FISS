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
NAME=ILT
INCREMENTAL_METHOD=ILT
STEPS_GLOBAL=5
TASK_NUM=6
EPOCHS_GLOBAL=`expr ${STEPS_GLOBAL} \* ${TASK_NUM}`

CLASS_RATIO=0.7
SAMPLE_RATIO2=1.0
BATCH_SIZE=12
EPOCHS_LOCAL=12

SEED=2023
echo ${EPOCHS_GLOBAL}

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/seed_${SEED}-ov/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} fl_main.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --base_weights --dataset ${DATASET} --name ${NAME} --task ${TASK} --incremental_method ${INCREMENTAL_METHOD} --class_ratio ${CLASS_RATIO} --sample_ratio2 ${SAMPLE_RATIO2} --batch_size ${BATCH_SIZE} --epochs_local ${EPOCHS_LOCAL} --steps_global ${STEPS_GLOBAL} --epochs_global ${EPOCHS_GLOBAL} --seed ${SEED} --opt_level O1 

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
