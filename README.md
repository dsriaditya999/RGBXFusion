# RGB-X Object Detection via Scene-Specific Fusion Modules

Multimodal deep sensor fusion has the potential to enable autonomous vehicles to visually understand their surrounding environments in all weather conditions. However, existing deep sensor fusion methods usually employ convoluted architectures with intermingled multimodal features, requiring large coregistered multimodal datasets for training. In this work, we present an efficient and modular RGB-X fusion network that can leverage and fuse pretrained single-modal models via scene-specific fusion modules, thereby enabling joint input-adaptive network architectures to be created using small, coregistered multimodal datasets. Our experiments demonstrate the superiority of our method compared to existing works on RGB-thermal and RGB-gated datasets, performing fusion using only a small amount of additional parameters.


## Setup


### Environment

Before starting, we will need Anaconda installed. Then create we can create new environment using:

```
conda create -n dsfusion python=3.9
```

To install requirements, we can use the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Toy Example
A few sample images from the FLIR_Aligned dataset are provided to enable users to quickly test out the training and validation functionalities without committing to downloading the full datasets. 

### Inference
TODO
### Validation
TODO
### Training
TODO

## Advanced

### Datasets

#### FLIR Aligned

Download the FLIR Aligned Dataset and put it in this structure:

```
├── FLIR_Aligned
│   ├── images_rgb_train
│   ├── images_rgb_test
│   ├── images_thermal_train
│   ├── images_thermal_test
│   ├── meta
```


#### M3FD Dataset

Download the FLIR Aligned Dataset and put it in this structure:

```
├── M3FD
│   ├── Ir
│   ├── Vis
│   ├── meta
```

#### STF Dataset

Download the FLIR Aligned Dataset and put it in this structure:

```
├── STF
│   ├── gated_full_acc_rect_aligned
│   ├── cam_stereo_left_rect_aligned
│   ├── meta
```

### Checkpoints

Various model checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1DXBIEsu799aDVCkwiGxQ5Y1bBcHUdrqv?usp=sharing).

## Training RGB Modality Checkpoint
TODO

## Training Thermal Modality Checkpoint
TODO

## Training Gated Modality Checkpoint
TODO

## Validating RGB Modality Checkpoint

To validate the provided Pretrained RGB checkpoint on FLIR Day Data (Test), run the following command:

```
python validate_fusion.py <dir of flir dataset> --dataset flir_aligned_day --init-fusion-head-weights rgb --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch rgb
```

## Validating Thermal Modality Checkpoint

To validate the provided Pretrained Thermal checkpoint on m3fd Night Data (Test), run the following command:

```
python validate_fusion.py <dir of m3fd dataset> --dataset m3fd_night --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --branch thermal
```

## Validating Gated Modality Checkpoint

To validate the provided Pretrained Gated checkpoint on STF Clear Data (Test), run the following command:

```
python validate_fusion.py <dir of STF dataset> --dataset stf_clear --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --classwise --split test --num-classes 4 --model efficientdetv2_dt --batch-size=8 --branch thermal 
```


## Training RGB-X Models

To train a RGB-Thermal Fusion (CBAM) on day data of FLIR Dataset, run the following command:

```
python train_fusion.py <dir of flir dataset> --dataset flir_aligned_day --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

To train a RGB-Thermal Fusion (CBAM) on overcast data of m3fd Dataset, run the following command:

```
python train_fusion.py <dir of m3fd dataset> --dataset m3fd_overcast --rgb-checkpoint-path <path to rgb checkpoint> --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

To train a RGB-Gated Fusion (CBAM) on full data of STF Dataset, run the following command:

```
python train_fusion.py <dir of STF dataset> --dataset stf_full --rgb-checkpoint-path <path to rgb checkpoint> --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --num-classes 4 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

## Validating RGB-X Models

To validate a trained (on night) RGB-Thermal Fusion (CBAM) on night data of FLIR Dataset, run the following command:

```
python validate_fusion.py <dir of flir dataset> --dataset flir_aligned_night --checkpoint <path to trained model> --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```

To validate a trained (on full) RGB-Thermal Fusion (CBAM) on challenge data of m3fd Dataset, run the following command:

```
python validate_fusion.py <dir of m3fd dataset> --dataset m3fd_challenge --checkpoint <path to trained model> --classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```


To validate a trained (on Fog Day) RGB-Gated Fusion (CBAM) on Snow Night of STF Dataset, run the following command:

```
python validate_fusion.py <dir of STF dataset> --dataset fog_day --checkpoint <path to trained model> --classwise --split test --num-classes 4 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```


## Training Classifier
TODO



## Scene-Adaptive Validation

To do a scene-adaptive validation on all data of FLIR Dataset, run the following command:

```
python validate_fusion_adaptive.py <dir of flir dataset> --dataset flir_aligned_full --num-scenes 3 --checkpoint <path to model trained on all flir data> --checkpoint-cls <path to classifier> --checkpoint-scenes <path to model trained on all flir data> <path to model trained on day flir data> <path to model trained on night flir data>  --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```

To do a scene-adaptive validation on all data of m3fd Dataset, run the following command:

```
python validate_all_att_cls.py <dir of m3fd dataset> --dataset m3fd_full --num-scenes 5 \
--checkpoint <path to model trained on all m3fd data> \
--checkpoint-cls <path to classifier> \
--checkpoint-scenes <path to model trained on all m3fd data> \
<path to model trained on day m3fd data> \
<path to model trained on night m3fd data> \
<path to model trained on overcast m3fd data> \
<path to model trained on challenge m3fd data> \
--split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```

To do a scene-adaptive validation on all data of STF Dataset, run the following command:

```
python validate_all_att_cls.py <dir of stf dataset> --dataset stf_full --num-scenes 7 \
--checkpoint <path to model trained on all stf data> \
--checkpoint-cls <path to classifier> \
--checkpoint-scenes <path to model trained on all stf data> \
<path to model trained on clear day stf data> \
<path to model trained on clear night stf data> \
<path to model trained on fog day stf data> \
<path to model trained on fog night stf data> \
<path to model trained on snow day stf data> \
<path to model trained on snow night stf data> \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```


To do a scene-adaptive validation on all clear data of STF Dataset, run the following command:

```
python validate_all_att_cls.py <dir of stf dataset> --dataset stf_clear --num-scenes 3 \
--checkpoint <path to model trained on all clear stf data> \
--checkpoint-cls <path to classifier> \
--checkpoint-scenes <path to model trained on all clear stf data> \
<path to model trained on clear day stf data> \
<path to model trained on clear night stf data> \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```



