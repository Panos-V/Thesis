#!/bin/bash

gpus=-1
checkpoint_root=checkpoint_test

img_size=32
batch_size=16
lr=1e-4
max_epochs=200

project_name=test
data_name=test
train=classifier

python new_main.py --gpu_ids ${gpus} --checkpoint_root ${checkpoint_root} \
    --img_size ${img_size} --batch_size ${batch_size} --lr ${lr} \
    --project_name ${project_name} \
    --data_name ${data_name} --train ${train} --max_epochs ${max_epochs}
