# wandb disabled
wandb enabled
wandb offline
    
CUDA_VISBLE_DEVICES=0 python train_FKD.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 100 \
    --model "ResNet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.15 \
    -j 4 --gradient-accumulation-steps 1 \
    -T 20 --sgd-lr 0.01 --adamw-lr 0.001 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_10_Recover_IPC_10 \
    --val-dir /path/to/ciaf-10/ \
    --fkd-path ../relabel/FKD_cutmix_fp16FKD_IPC_10
