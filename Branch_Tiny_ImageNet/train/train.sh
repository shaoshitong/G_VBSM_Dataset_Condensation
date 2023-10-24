# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python train_FKD.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 100 \
    --model "ResNet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.1 \
    -j 4 --gradient-accumulation-steps 2 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.0005 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_Tiny_ImageNet_Recover_IPC_50 \
    --val-dir /path/to/tiny-imagenet/ \
    --fkd-path ../relabel/FKD_cutmix_fp16FKD_IPC_50