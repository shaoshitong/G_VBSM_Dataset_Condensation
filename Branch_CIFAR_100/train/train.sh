# wandb disabled
wandb enabled
wandb offline

CUDA_VISBLE_DEVICES=0 python train_FKD.py \
    --wandb-project 'final_cw128_fkd' \
    --batch-size 64 \
    --model "ConvNetW128" \
    --cos --loss-type "mse_gt" --ce-weight 0.15 \
    -j 4 --gradient-accumulation-steps 1 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_cw128_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_100_Recover_IPC_10 \
    --val-dir /path/to/cifar-100/ \
    --fkd-path ../relabel/FKD_cutmix_fp16FKD_IPC_10 # model in [ConvNetW128, ResNet18]