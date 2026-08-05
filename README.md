# AS-Mono

This repository provides the official PyTorch implementation of **AS-Mono: Adaptive Structural Distillation for Lightweight Monocular Depth Estimation on Edge Devices**.

The codebase supports training, evaluation, and inference for monocular depth estimation models described in our paper.

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

## ⚙️Setup

Detailed environment requirements are provided in `environment.yml`, including the Python version, CUDA-compatible PyTorch dependencies, and other required packages for reproducing our experiments.

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 pillow==9.5.0 imageio==2.35.1

pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8

pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image

pip install mmcv-full==1.3.0 mmsegmentation==0.11.0  

pip install timm einops IPython
```
We ran our experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.8 and Ubuntu 20.04. 

Note that our code is built based on [Monodepth2](https://github.com/nianticlabs/monodepth2). 

## 

## 💾KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.



You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.

## ⏳Training

Pre-trained MonoViT weights are available at [here](https://github.com/zxcqlf/MonoViT) 

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

**Monocular training:**

```shell
python train.py --model_name model_name --num_layers 18 --encoder_mobilevit xs --decoder_channel_scale [200,100,50] 
```

The encoder_mobilevit means the backbone network of MobileViTv1

| Name |    Encoder    |
| :--: | :-----------: |
|  xs  | MobileViT_xs  |

The decoder_channel_scale means

| decoder channel scale | decoder channels for each stage |
| :-------------------: | :-----------------------------: |
|          2          |     {16, 32, 64, 128, 256}      |
|          1          |      {8, 16, 32, 64, 128}       |
|          0.5          |       {4, 8, 16, 32, 64}        |




## 📊KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
We assume that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:

```shell
python evaluate_depth_KITTI.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --decoder_channel_scale 100 --encoder_mobilevit xs --eval_mono
```


## 📊Make3D evaluation
```shell
python evaluate_depth_Make3D.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```


## 📊UAV evaluation
```shell
python UAV/evaluate_depth_UAV.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```

## 💾Latency Evaluation
The latency is averaged over 300 runs.

```shell
python evaluate_latency.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```

## Model Parameters and FLOPs Evaluation
```shell
python evaluate_flops_params.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```

## Boundary-Sensitive Metrics Evaluation
```shell
python evaluate_boundary_metrics.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```

## Latency and Throughput Testing on UAV

Note that `batch_size` should be set to 1 when testing latency, and set to 16 when testing throughput.
 
Then run the following command:

```shell
python UAV/evaluate_latency_throughput_UAV.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ \
	--height 192 --width 640 \
	--encoder_mobilevit xs --decoder_channel_scale 100
```


## Pretrained weights

The pretrained weights (MonoViT_M_640x192 and MPViT-small ) are not included in this repository due to their large file size. Please download them manually from the links below and place them in the following paths: 


```text 

# MobileViT-v1 pretrained weights
pretrained_weight/mobilevitv1/ 

# MonoViT_M_640x192
pretrained_weight/MonoViT_M_640x192/ 

# MPViT-small pretrained weights
pretrained_weight/mpvit_small.pth        

```
This download link is provided by the official MonoViT GitHub repository.  
[MonoViT weights](https://drive.google.com/drive/folders/1VWDPuqiMPDD2P--Oka-yJgh8z7ouCX4D?usp=sharing)

This download link is provided by the official MPViT GitHub repository.

[MPViT-small weights](https://dl.dropboxusercontent.com/s/y3dnmmy8h4npz7a/mpvit_small.pth)


## UAV Dataset Samples, Calibration, and Metadata
 
To improve transparency and reproducibility, we provide representative UAV RGB-D samples, calibration and alignment documentation, metadata, and the evaluation protocol in the `UAV/` directory.
 
The UAV-related files include:
 
- `UAV/MODEL_CARD.md`
- `UAV/calibration_and_alignment.md`
- `UAV/calibration.json`
- `UAV/metadata.json`
- Representative RGB frames
- Corresponding visualized depth maps
 

## Model Card for AS-Mono

### Model Overview

AS-Mono is a lightweight self-supervised monocular depth estimation framework designed for depth prediction on edge devices. It uses adaptive multi-order structural knowledge distillation to improve boundary preservation without adding inference-time cost.

### Model Details

- **Model name:**  AS-Mono
- **Task:** Self-supervised monocular depth estimation
- **Input:** Single RGB image
- **Output:** Depth map
- **Student network:** MobileViT-based lightweight encoder-decoder
- **Teacher network:** MonoViT
- **Training-only components:** Frozen teacher network and adaptive structural knowledge distillation module
- **Inference-time components:** Student DepthNet only
- **Default input resolution:** 640 × 192 for KITTI evaluation
- **Training paradigm:** Self-supervised monocular training using image reconstruction losses and teacher-guided adaptive structural distillation

### Intended Use

AS-Mono is intended for research and development in lightweight monocular depth estimation. Potential application scenarios include:
 
- Edge-device visual perception
- UAV perception
- Mobile robot navigation
- Real-time 3D scene understanding
- AR-assisted perception
- Resource-constrained visual computing systems


### Out-of-Scope Use

AS-Mono is not intended to be used as the sole perception module in safety-critical autonomous systems. Before deployment in real-world safety-sensitive scenarios, additional validation, sensor fusion, uncertainty estimation, and fail-safe mechanisms are required.


### Training Data

AS-Mono is trained on the KITTI dataset using the standard Eigen split protocol for self-supervised monocular depth estimation. The model is trained without using ground-truth depth labels during training.


### Evaluation Data

AS-Mono is evaluated on:

- KITTI Eigen split
- Make3D dataset
- Self-collected UAV RGB-D dataset

### Evaluation Protocol

For KITTI evaluation, we follow the standard monocular depth estimation protocol:
 
- Median scaling is applied during evaluation.
- Eigen crop is used.
- The maximum depth is capped at 80 meters.
- Standard depth estimation metrics are reported, including:
    - Abs Rel
    - Sq Rel
    - RMSE
    - RMSE log
    - δ < 1.25
    - δ < 1.25²
    - δ < 1.25³

For Make3D evaluation, we follow the standard cross-dataset monocular depth estimation protocol:
 
- All models are trained on KITTI and directly evaluated on Make3D without fine-tuning.
- Center-cropping is applied to the input images.
- Images are resized to 640 × 192 for evaluation.
- Median scaling is applied during evaluation.
- The Make3D crop is used.
- The maximum depth is capped at 70 meters.
- Standard depth estimation metrics are reported, including:
    - Abs Rel
    - Sq Rel
    - RMSE
    - RMSE log


For the self-collected UAV RGB-D dataset evaluation, we follow a consistent real-world generalization protocol:
 
- All models are trained on KITTI and directly evaluated on the self-collected UAV RGB-D dataset without fine-tuning.
- Baseline methods are evaluated using their officially released trained weights.
- The RGB and depth streams are spatially aligned using the intrinsic and extrinsic calibration parameters and the alignment procedure provided by the camera SDK.
- All input RGB images are resized to 640 × 192 for evaluation.
- Depth values outside the predefined valid range of 0.1–10 m are excluded from metric computation.
- Missing or invalid depth measurements are excluded from metric computation.
- All compared methods are evaluated using the same valid-depth mask, input resolution, preprocessing procedure, and evaluation code.
- Standard depth estimation metrics are reported, including:
    - Abs Rel
    - Sq Rel
    - RMSE
    - RMSE log
    - δ < 1.25
    - δ < 1.25²
    - δ < 1.25³

### Quantitative Results

Detailed quantitative results on KITTI, Make3D, and the UAV dataset are provided in the paper.


### Deployment Evaluation

AS-Mono is evaluated on both desktop GPU and edge-device platforms, including:
 
- NVIDIA GeForce RTX 4090
- NVIDIA Jetson Orin NX
 

The reported deployment metrics include:
 
- Number of parameters
- FLOPs
- Batch-size-1 inference latency
- Batch-size-16 inference time
- Edge-device runtime performance
 

Please refer to the paper and benchmarking scripts in the repository for detailed numerical results and hardware settings.



### Model Weights Availability

Due to the large file size, pretrained weights are not directly stored in this GitHub repository. Please download the required pretrained weights from the links provided in the README and place them in the specified directories.

- MobileViT-v1 pretrained weights
- MonoViT pretrained weights
- MPViT-small pretrained weights

 
### Reproducibility
 
To support reproducibility, the repository provides:
 
- Source code
- Training scripts
- Evaluation scripts
- Configuration files
- Preprocessing tools
- Instructions for downloading required pretrained weights
 

Please follow the README for environment setup, dataset preparation, weight placement, training, evaluation, and benchmarking instructions.

### Limitations

AS-Mono inherits common limitations of self-supervised monocular depth estimation, including:
 
- Scale ambiguity in monocular depth prediction
- Sensitivity to dynamic objects
- Sensitivity to motion blur
- Reduced robustness in low-light scenes
- Difficulty with reflective, transparent, or textureless surfaces

Therefore, users should carefully validate the model before applying it to new environments or deployment scenarios.
 
### Ethical and Safety Considerations
 
AS-Mono is designed for research in visual perception and depth estimation. It should not be used as the only source of environmental understanding in systems where incorrect depth predictions may cause physical harm, financial loss, or safety risks.

For real-world deployment, we recommend combining AS-Mono with additional sensors, uncertainty estimation, and system-level safety checks.


### License

The code is released under the license specified in the `LICENSE` file of this repository.

## Acknowledgement

We would like to thank the authors of the following works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MViTDepth](https://github.com/mengmengbi/MViTDepth)

[MobileViTv1](https://github.com/apple/ml-cvnets) 

[MonoViT](https://github.com/zxcqlf/MonoViT)

[MPViT](https://github.com/youngwanLEE/MPViT)


