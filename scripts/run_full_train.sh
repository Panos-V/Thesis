#!/bin/bash

gpus=0
checkpoint_root=checkpoint_test

img_size=32
batch_size=128
lr=1e-2
max_epochs=200

project_name=First_full_train
data_name=Fitzpatrick17k
train=vqvae

python new_main.py --gpu_ids ${gpus} --checkpoint_root ${checkpoint_root} \
    --img_size ${img_size} --batch_size ${batch_size} --lr ${lr} \
    --project_name ${project_name} \
    --data_name ${data_name} --train ${train} --max_epochs ${max_epochs}
