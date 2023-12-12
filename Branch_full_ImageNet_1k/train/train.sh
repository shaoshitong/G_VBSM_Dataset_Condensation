# wandb disabled
wandb enabled
wandb offline

CUDA_VISBLE_DEVICES=0,1 python train_FKD_parallel.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 1024 \
    --model "resnet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.1 \
    -j 4 --gradient-accumulation-steps 1 \ # 1 for resnet18, 4 for resnet50 and resnet101
    -T 20 --gpu-id 0,1 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_ImageNet_1k_Recover_IPC_10 \
    --val-dir /path/to/imagenet-1k/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16FKD_cutmix_fp16