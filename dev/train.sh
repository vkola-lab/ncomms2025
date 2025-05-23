#!/bin/bash -l

# run this script from adrd_tool/

conda activate /projectnb/vkolagrp/skowshik/conda_envs/adrd
pip install .

# install the package
# cd adrd_tool
# pip install -e .

# define the variables
prefix="."
train_path="${prefix}/pseudodata/synthetic_train.csv"
vld_path="${prefix}/pseudodata/synthetic_val.csv"


# Note for setting the flags
# 1. If training without MRIs
# img_net="NonImg"
# img_mode = -1
# 2. if training with MRIs
# img_net: [SwinUNETR]
# img_mode = 0
# 3. if training with MRI embeddings
# img_net: [SwinUNETREMB]
# img_mode = 1

# Without using image embeddings
img_net="NonImg"
img_mode=-1

# Using image embeddings
# img_net="SwinUNETREMB"
# img_mode=1

# Stage 1
cnf_file="${prefix}/dev/data/toml_files/stage_1.toml"
ckpt_path="./dev/ckpt/model_stage_1.pt"

# run train.py 
python dev/train.py --train_path $train_path --vld_path $vld_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
                    --num_epochs 128 --batch_size 128 --lr 1e-3 --gamma 2 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" \
                    --ckpt_path $ckpt_path --cnf_file ${cnf_file} --train_path ${train_path} --vld_path ${vld_path}  \
                    --fusion_stage middle --imgnet_layers 2 --weight_decay 0.01 --n_splits 1 --stage 1 --save_intermediate_ckpts --early_stop_threshold 15 --transfer_epoch 15 --device "cuda" --fine_tune  #--wandb_project "Project" --wandb 


# Stage 2
# cnf_file="${prefix}/dev/data/toml_files/stage_2.toml"
# ckpt_path="./dev/ckpt/model_stage_2.pt"

# # run train.py 
# python dev/train.py --train_path $train_path --vld_path $vld_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
#                     --num_epochs 128 --batch_size 128 --lr 1e-4 --gamma 2 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" \
#                     --ckpt_path $ckpt_path --cnf_file ${cnf_file} --train_path ${train_path} --vld_path ${vld_path}  \
#                     --fusion_stage middle --imgnet_layers 2 --weight_decay 0.005 --n_splits 1 --stage 2 --save_intermediate_ckpts --early_stop_threshold 30 --wandb_project "Project" --wandb --fine_tune