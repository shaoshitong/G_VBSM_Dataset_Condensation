# GV-BSM


## Install

Our code can be easily run, you only need install following packages:
```bash
pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
pip install einops timm kornia tqdm wandb prefetch_generator scipy
```
Note that almost all versions of torch will work, you do not necessarily install torch2.0.


## Our code based on SRe2L, MTT and Good-DA-in-KD, please cite:

```
@article{yin2023squeeze,
	title = {Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective},
	author = {Yin, Zeyuan and Xing, Eric and Shen, Zhiqiang},
	journal = {arXiv preprint arXiv:2306.13092},
	year = {2023}
}


@inproceedings{wang2022what,
  author = {Huan Wang and Suhas Lohit and Michael Jones and Yun Fu},
  title = {What Makes a "Good" Data Augmentation in Knowledge Distillation -- A Statistical Perspective},
  booktitle = {NeurIPS},
  year = {2022}
}

@inproceedings{
cazenavette2022distillation,
title={Dataset Distillation by Matching Training Trajectories},
author={George Cazenavette and Tongzhou Wang and Antonio Torralba and Alexei A. Efros and Jun-Yan Zhu},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```

## Run
The implementation of different datasets are encapsulated in different directory: [CIFAR-10](./Branch_CIFAR_10), [CIFAR-100](./Branch_CIFAR_100), [Tiny-ImageNet](./Branch_Tiny_ImageNet) and [ImageNet-1k](./Branch_full_ImageNet_1k). After entering these directories, you can run:
```bash
cd squeeze
bash ./squeeze.sh # Only for CIFAR-10/CIFAR-100, Tiny-ImageNet.

cd recover
bash ./recover.sh

cd ../relabel
bash ./relabel.sh

cd ../train
bash ./train.sh
```