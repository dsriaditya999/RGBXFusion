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

### Datasets

Download the FLIR Aligned Dataset and put it in this structure:

```
├── FLIR_Aligned
│   ├── images_rgb_train
│   ├── images_rgb_test
│   ├── images_thermal_train
│   ├── images_thermal_test
│   ├── meta
```

## Validating RGB Modality Checkpoint

To validate the provided Pretrained RGB checkpoint on Day Data (Test), run the following command:

```
python validate_fusion.py <dir of flir dataset> --dataset flir_aligned_day --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch rgb
```

## Validating Thermal Modality Checkpoint

To validate the provided Pretrained RGB checkpoint on Night Data (Test), run the following command:

```
python validate_fusion.py <dir of flir dataset> --dataset flir_aligned_night --thermal-checkpoint-path <path to thermal checkpoint> --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch thermal
```


## Training an RGB-T Model

To train a RGB-T Fusion (CBAM) on day data of FLIR Dataset, run the following command:

```
python train_fusion.py <dir of flir dataset> --dataset flir_aligned_day --thermal-checkpoint-path <path to thermal checkpoint> --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

## Validating an RGB-T Model

To validate a trained (on night) RGB-T Fusion (CBAM) on night data of FLIR Dataset, run the following command:

```
python validate_fusion.py <dir of flir dataset> --dataset flir_aligned_night --checkpoint <path to trained model> --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```



## Scene-Adaptive Validation

To do a scene-adaptive validation on all data of FLIR Dataset, run the following command:

```
python validate_fusion_adaptive.py <dir of flir dataset> --dataset flir_aligned_full --num-scenes 3 --checkpoint <path to model trained on all flir data> --checkpoint-cls <path to classifier> --checkpoint-scenes <path to model trained on all flir data> <path to model trained on day flir data> <path to model trained on night flir data>  --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```




