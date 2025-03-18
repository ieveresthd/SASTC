# SASTC

Official implementation of "[Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation](https://arxiv.org/abs/2501.08049)" (**AAAI2025**).

By Di Hong & Yueming Wang.

*This study focuses on mitigating performance degradation due to spatio-temporal semantic mismatches and negative regularization in conventional ANN-to-SNN knowledge distillation methods.We propose a self-attentive mechanism to learn layer association weights across different time steps, enabling semantically aligned knowledge transfer.*

![overview](/Users/ieverest/Research/GitHub/SASTC/fig/SASTC_overview.jpg)

## Requirements

In order to run this project you will need:

- Python3 >= 3.11
- PyTorch>=2.3.0
```
conda create --name new_env --file requirements.txt
```
```
pip install -r requirements.txt
```

The code is stored in five different file folders for static (CIFAR-10, CIFAR-100, ImageNet) and neuromorphic (DVS-Gesture and DVS-CIFAR10) tasks. It supports single GPU or multiple GPUs.

## ANN (teacher) Training

Train on the CIFAR-10:

```
python3 train_teacher.py --arch wrn_28_4 -lr 0.1 --weight-decay 5e-4 --log --save --gpu_id 0
```



## SNN Training (ANN-to-SNN Distillation)

Train on the CIFAR-10:

```
python3 main_distill_distribute.py --batch-size 16 --T 7 --beta 800 --distill_type SASTC --snn_arch wrn_16_2 --ann_arch resnet19 --ta_path result/ann/resnet19/ann_model_best.pth.tar --gpu_id 0,1,2,3 --address tcp://127.0.0.1:2379 --log --save
```



## Citation

If you find our work is useful for your research, please kindly cite our paper:

```
@article{hong2025self,
  title={Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation},
  author={Hong, Di and Wang, Yueming},
  journal={arXiv preprint arXiv:2501.08049},
  year={2025}
}
```