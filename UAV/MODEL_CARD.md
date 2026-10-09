# Model Card for AS-Mono

## Model Overview

AS-Mono is a lightweight self-supervised monocular depth estimation framework designed for depth prediction on edge devices. It uses adaptive multi-order structural knowledge distillation to improve boundary preservation without adding inference-time cost.

## Model Details

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

## Intended Use

AS-Mono is intended for research and development in lightweight monocular depth estimation. Potential application scenarios include:
 
- Edge-device visual perception
- UAV perception
- Mobile robot navigation
- Real-time 3D scene understanding
- AR-assisted perception
- Resource-constrained visual computing systems


## Out-of-Scope Use

AS-Mono is not intended to be used as the sole perception module in safety-critical autonomous systems. Before deployment in real-world safety-sensitive scenarios, additional validation, sensor fusion, uncertainty estimation, and fail-safe mechanisms are required.


## Training Data

AS-Mono is trained on the KITTI dataset using the standard Eigen split protocol for self-supervised monocular depth estimation. The model is trained without using ground-truth depth labels during training.


## Evaluation Data

AS-Mono is evaluated on:

- KITTI Eigen split
- Make3D dataset
- Self-collected UAV RGB-D dataset

## Evaluation Protocol

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

## Quantitative Results

Detailed quantitative results on KITTI, Make3D, and the UAV dataset are provided in the paper.


## Deployment Evaluation

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



## Model Weights Availability

Due to the large file size, pretrained weights are not directly stored in this GitHub repository. Please download the required pretrained weights from the links provided in the README and place them in the specified directories.

- MobileViT-v1 pretrained weights
- MonoViT pretrained weights
- MPViT-small pretrained weights

 
## Reproducibility
 
To support reproducibility, the repository provides:
 
- Source code
- Training scripts
- Evaluation scripts
- Configuration files
- Preprocessing tools
- Instructions for downloading required pretrained weights
 

Please follow the README for environment setup, dataset preparation, weight placement, training, evaluation, and benchmarking instructions.

## Limitations

AS-Mono inherits common limitations of self-supervised monocular depth estimation, including:
 
- Scale ambiguity in monocular depth prediction
- Sensitivity to dynamic objects
- Sensitivity to motion blur
- Reduced robustness in low-light scenes
- Difficulty with reflective, transparent, or textureless surfaces

Therefore, users should carefully validate the model before applying it to new environments or deployment scenarios.
 
## Ethical and Safety Considerations
 
AS-Mono is designed for research in visual perception and depth estimation. It should not be used as the only source of environmental understanding in systems where incorrect depth predictions may cause physical harm, financial loss, or safety risks.

For real-world deployment, we recommend combining AS-Mono with additional sensors, uncertainty estimation, and system-level safety checks.


## License

The code is released under the license specified in the `LICENSE` file of this repository.