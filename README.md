# MOD-UV: Learning Mobile Object Detectors from Unlabeled Videos
### [Project Page](https://mod-uv.github.io) | [Paper](https://arxiv.org/pdf/2405.14841.pdf) | [arXiv](https://arxiv.org/abs/2405.14841)

Official PyTorch implementation for the ECCV 2024 paper: "MOD-UV: Learning Mobile Object Detectors from Unlabeled Videos".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

![](assets/teaser.jpg)

## Installation
The code is tested with `python=3.7`, `torch==1.12.1` and `torchvision==0.13.1` on a NVIDIA A6000 GPU.
```
conda create -n moduv python=3.7
conda activate moduv
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install matplotlib wandb opencv-python tqdm gdown scikit-image scikit-learn imageio pycocotools einops
```

Please first use `scripts/0_download_ckpts.sh` for downloading the relevant checkpoints.
```
bash scripts/0_download_ckpts.sh
```
**Note**: To access the trained model checkpoints on Waymo Open, please fill out the [Google Form](https://forms.gle/gy2SpDSegMLDkm2o7), and [raise an issue](https://github.com/YihongSun/MOD-UV/issues/new) if we don't get back to you in three days. Please note that Waymo open dataset is under strict non-commercial license so we are not allowed to share the model with you if it will used for any profit-oriented activities.

## Quick Demo and Inference

Once you have access to the model checkpoints, please use [demo.ipynb](demo.ipynb) for an example inference of our method at each phase of self-training and pseudo-label generation.

Also, [demo.py](demo.py) generates the same visualizations as found in the qualitative figures (saved in `./out/`):
```
python3 demo.py --load_ckpt ./ckpts/moduv_final.pth --vis_conf_thrd 0.5 --input data/ --out ./out/
```

_Try your own images_: to run our model locally on any input image(s):
```
python3 demo.py --load_ckpt <PATH/TO/MODEL> --vis_conf_thrd <CONFIDENCE_THRD> --input <PATH/TO/IMAGE(S)> --out <PATH/TO/OUTPUT_DIR>
```

## Training

### Stage 0: Setting up Waymo Dataset

To obtain the Waymo training sequences, please refer to the [official website](https://waymo.com/open/) for downloading the Waymo Open Dataset.
Once downloaded and extracted, please format the training images as following. 
- Please refer to the data processing pipeline found in [Dynamo-Depth](https://github.com/YihongSun/Dynamo-Depth) if needed. The extracted data should be match with the images in `MOD-UV/data/waymo/`.

```
MOD-UV/data/waymo/
  |-- train/
    |-- segment-10017090168044687777_6380_000_6400_000/
      |-- 000000.jpg
      |-- 000001.jpg
      |-- ...
      |-- 000197.jpg
    |-- segment-10023947602400723454_1120_000_1140_000/
    |-- ...
    |-- segment-9985243312780923024_3049_720_3069_720/
  |-- val
    |-- segment-10203656353524179475_7625_000_7645_000/
      |-- 000000.jpg
      |-- 000001.jpg
      |-- ...
      |-- 000197.jpg
    |-- segment-1024360143612057520_3580_000_3600_000/
    |-- ...
    |-- segment-967082162553397800_5102_900_5122_900/
```

### Stage 1: Computing Initial Pseudo-Labels $L^{(0)}$

To obtain the initial pseudo-labels $L^{(0)}$ for training, please run the following bash script to download the precomputed initial pseudo-labels. (Filling out the [Google Form](https://forms.gle/gy2SpDSegMLDkm2o7) for checkpoints would also grant access for the initial pseudo-labels.)
```
bash scripts/1_download_pseudo_labels.sh
```
For details in the pseudo-label computations, please refer to the example in [demo.ipynb](demo.ipynb) and implementation in [moduv/pseudo_labels/init_labels.py](moduv/pseudo_labels/init_labels.py).

### Stage 2: Computing Pseudo-Labels after _Moving2Mobile_ $L^{(1)}$

To obtain the pseudo-labels after _Moving2Mobile_ $L^{(1)}$ for training, please run the following bash script.
```
bash scripts/2_moving2mobile.sh
```
For details in the pseudo-label computations, please refer to the example in [demo.ipynb](demo.ipynb) and implementation in [moduv/pseudo_labels/save_pred.py](moduv/pseudo_labels/save_pred.py).

### Stage 3: Computing Pseudo-Labels after Large2Small $L^{(2)}$

To obtain the pseudo-labels after _Large2Small_ $L^{(2)}$ for training, please run the following bash script.
```
bash scripts/3_large2small.sh
```
For details in the pseudo-label computations, please refer to the example in [demo.ipynb](demo.ipynb) and implementation in [moduv/pseudo_labels/save_agg_pred.py](moduv/pseudo_labels/save_agg_pred.py).

### Stage 4: Final Self-Training Round with $L^{(2)}$

To obtain the final detector, please run the following bash script.
```
bash scripts/4_final.sh
```

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{sun2024moduv,
  title={MOD-UV: Learning Mobile Object Detectors from Unlabeled Videos}, 
  author={Yihong Sun and Bharath Hariharan},
  booktitle={ECCV},
  year={2024}
}
```
and related work in learning monocular depth and motion segmentation from unlabeled videos:
```
@inproceedings{sun2023dynamodepth,
  title={Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes},
  author={Yihong Sun and Bharath Hariharan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
