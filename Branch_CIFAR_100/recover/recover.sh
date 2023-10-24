CUDA_VISIBLE_DEVICES=0,1 python data_synthesis_with_svd_with_db_with_all_statistic.py \
    --arch-name "resnet18" \
    --exp-name "GVBSM_CIFAR_100_Recover_IPC_10" \
    --batch-size 100 \
    --lr 0.05 \
    --ipc-number 10 \
    --iteration 4000 \
    --train-data-path /path/to/cifar-100/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1