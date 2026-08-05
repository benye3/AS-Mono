# Calibration and RGB-D Alignment

## Sensor Setup

The self-collected UAV RGB-D dataset was captured using an Intel RealSense D435i camera rigidly mounted on the UAV platform. The camera was used to collect synchronized RGB and depth streams during indoor UAV flight.

Both RGB images and aligned depth maps have an original resolution of 640 × 480.

## RGB-D Alignment

The RGB and depth streams were spatially aligned using the intrinsic and extrinsic calibration parameters provided by the RealSense SDK. Specifically, the depth stream was aligned to the RGB camera coordinate frame using the official RealSense SDK alignment procedure.

The released depth maps have already been spatially aligned with the corresponding RGB images. Therefore, users can directly use the provided RGB-depth pairs for evaluation without performing additional RGB-D registration.

## Calibration Parameters

The camera calibration parameters are provided in `calibration.json`, including:

- RGB camera intrinsics
- Depth camera intrinsics
- Depth-to-RGB extrinsics
- Depth scale
- Image resolution

The calibration parameters were obtained from the RealSense SDK during data acquisition.

## Depth Format
 
Depth values are stored in meters. During evaluation, only valid depth pixels are used for metric computation.
 
A depth pixel is considered valid if:
 
- The depth value is available.
- The depth value is finite.
- The depth value lies within the predefined evaluation range of 0.1–10 m.
 
Depth values outside the range of 0.1–10 m, as well as missing or invalid depth measurements, are excluded from metric computation.
 
## Evaluation Alignment Policy
 
All compared methods are evaluated using the same:
 
- RGB input resolution
- Aligned depth maps
- Valid-depth mask
- Preprocessing procedure
- Evaluation code
 
This ensures a fair comparison across all evaluated methods.
