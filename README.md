# RGB-X Object Detection via Scene-Specific Fusion Modules

Multimodal deep sensor fusion has the potential to enable autonomous vehicles to visually understand their surrounding environments in all weather conditions. However, existing deep sensor fusion methods usually employ convoluted architectures with intermingled multimodal features, requiring large coregistered multimodal datasets for training. In this work, we present an efficient and modular RGB-X fusion network that can leverage and fuse pretrained single-modal models via scene-specific fusion modules, thereby enabling joint input-adaptive network architectures to be created using small, coregistered multimodal datasets. Our experiments demonstrate the superiority of our method compared to existing works on RGB-thermal and RGB-gated datasets, performing fusion using only a small amount of additional parameters.

## Setup
### Environment
Before starting, install [anaconda](https://docs.conda.io/en/latest/miniconda.html#installing) in order to create a virtual environment and create a new Python 3.9 environment using:

```
conda create -n dsfusion python=3.9
```
Note: our code has been tested with Python 3.9. It is not garuanteed to work with other other versions. 

 Activate the environment and install the necessary packages via:

```
 conda activate dsfusion
 pip install -r requirements.txt
 ```

### Checkpoints
Download model checkpoints from [here](https://drive.google.com/drive/folders/1-RHJ2e6LfvPb0UTC-hkHiEts4W2Dz1up?usp=sharing), and directly extract it into the `Checkpoints` folder. The folder should have the following structure:
```
Checkpoints
├── FLIR_Aligned
│   ├── Classifier
│   ├── Fusion_Models
│   │   ├── Day
│   │   ├── Night
│   │   └── Full
│   └── Single_Modality_Models
├── M3FD
│   ├── Classifier
│   ├── Fusion_Models
│   │   ├── Day
│   │   ├── Night
│   │   ├── Overcast
│   │   ├── Challenge
│   │   └── Full
│   └── Single_Modality_Models
└── STF
    ├── Classifiers
    ├── Fusion_Models
    │   ├── All_Trained
    │   │   ├── Clear_Day
    │   │   ├── Clear_Night
    │   │   ├── Fog_Day
    │   │   ├── Fog_Night
    │   │   ├── Snow_Day
    │   │   ├── Snow_Night
    │   │   └── Full
    │   └── Clear_Trained
    │       ├── Clear_Day
    │       ├── Clear_Night
    │       └── Clear
    └── Single_Modality_Models
        ├── All_Trained
        └── Clear_Trained
```

### Datasets
Download the FLIR Aligned Dataset from [here](https://drive.google.com/drive/folders/18XmdzKj0sGOFt0r4LmwMo9TsVNpyKEzT?usp=sharing), and extract it into the `Datasets` folder. The folder should have the following structure:
```
Datasets
├── FLIR_Aligned
│   ├── images_rgb_train
│   ├── images_rgb_test
│   ├── images_thermal_train
│   ├── images_thermal_test
│   ├── meta
├── M3FD
│   ├── Ir
│   ├── Vis
│   ├── meta
└── STF
    ├── gated_full_acc_rect_aligned
    ├── cam_stereo_left_rect_aligned
    └── meta
```

## Toy Example
A few sample images from the FLIR_Aligned dataset are provided to enable users to quickly test out the training and validation functionalities without committing to downloading the full datasets. 

#### Setup
Download FLIR checkpoint model files from [here](https://drive.google.com/drive/folders/1-RHJ2e6LfvPb0UTC-hkHiEts4W2Dz1up?usp=sharing) and put it into the `Checkpoints` folder. See instructions of checkpoints above.

Note: Make sure that you are in the `RGBXFusion` directory when running the following commands.
#### Validating an RGB-T Model
To validate a trained (on day) RGB-Thermal Fusion (CBAM) model on day data of toy FLIR Dataset, run the following command:
```
python validate_fusion.py toy/data --dataset flir_aligned_day --checkpoint "Checkpoints/FLIR_Aligned/Fusion_Models/Day/model_best.pth.tar" --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```

#### Scene Adaptive Validation
To do a scene-adaptive validation on all data of toy FLIR Dataset, run the following command:
```
python validate_fusion_adaptive.py toy/data --dataset flir_aligned_full --num-scenes 3 --checkpoint "Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar" --checkpoint-cls "Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar" --checkpoint-scenes "Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar" "Checkpoints/FLIR_Aligned/Fusion_Models/Day/model_best.pth.tar" "Checkpoints/FLIR_Aligned/Fusion_Models/Night/model_best.pth.tar"  --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam
```

#### Training an RGB-T Model
To train a RGB-Thermal Fusion (CBAM) model on night data of toy FLIR Dataset, run the following command:
```
python train_fusion.py toy/data --dataset flir_aligned_night --thermal-checkpoint-path "Checkpoints/FLIR_Aligned/Single_Modality_Models/flir_thermal_backbone.pth.tar" --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

## Training
#### Training Single Modality Backbone and Detector
We use single modality backbones extracted from trained object detectors.
* To train the FLIR rgb single modality object detector, do
```
bash bash/train_single.sh rgb
```

* To train the FLIR thermal single modality object detector, do
```
bash bash/train_single.sh thermal
```
Look inside the bash scripts and `train_single.py` file to better understand and set custom training parameters best suited for your hardware setup.

#### Training RGB-X Models
* To train a RGB-Thermal Fusion (CBAM) model on day data of FLIR Dataset, run the following command:
```
python train_fusion.py Datasets/FLIR_Aligned --dataset flir_aligned_day --thermal-checkpoint-path Checkpoints/FLIR_Aligned/Single_Modality_Models/flir_thermal_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

* To train a RGB-Thermal Fusion (CBAM) model on overcast data of m3fd Dataset, run the following command:
```
python train_fusion.py Datasets/M3FD --dataset m3fd_overcast --rgb-checkpoint-path Checkpoints/M3FD/Single_Modality_Models/m3fd_rgb_backbone.pth.tar --thermal-checkpoint-path Checkpoints/M3FD/Single_Modality_Models/m3fd_thermal_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

* To train a RGB-Gated Fusion (CBAM) model on full data of STF Dataset, run the following command:
```
python train_fusion.py Datasets/STF --dataset stf_full --rgb-checkpoint-path Checkpoints/STF/Single_Modality_Models/All_Trained/stf_rgb_backbone.pth.tar --thermal-checkpoint-path Checkpoints/STF/Single_Modality_Models/All_Trained/stf_gated_backbone.pth.tar --init-fusion-head-weights thermal --num-classes 4 --model efficientdetv2_dt --batch-size=8 --epochs=50 --branch fusion --freeze-layer fusion_cbam --att_type cbam
```

## Validation
#### Validate Single Modality Checkpoins
* To validate the provided Pretrained Thermal checkpoint on m3fd Night Data (Test), run the following command:  
```
python validate_fusion.py Datasets/M3FD --dataset m3fd_night --thermal-checkpoint-path Checkpoints/M3FD/Single_Modality_Models/m3fd_thermal_backbone.pth.tar --init-fusion-head-weights thermal --classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --branch thermal
```

* To validate the provided Pretrained Gated checkpoint on STF Clear Data (Test), run the following command:   
```
python validate_fusion.py Datasets/STF --dataset stf_clear --thermal-checkpoint-path Checkpoints/STF/Single_Modality_Models/Clear_Trained/stf_gated_backbone.pth.tar --init-fusion-head-weights thermal --classwise --split test --num-classes 4 --model efficientdetv2_dt --batch-size=8 --branch thermal 
```

#### Validating RGB-X Models
* To validate a trained (on night) RGB-Thermal Fusion (CBAM) model on night data of FLIR Dataset, run the following command:
```
python validate_fusion.py Datasets/FLIR_Aligned --dataset flir_aligned_night --checkpoint Checkpoints/FLIR_Aligned/Fusion_Models/Night/model_best.pth.tar --classwise --split test --num-classes 90 --rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```

* To validate a trained (on full) RGB-Thermal Fusion (CBAM) model on challenge data of m3fd Dataset, run the following command:
```
python validate_fusion.py Datasets/M3FD --dataset m3fd_challenge --checkpoint Checkpoints/M3FD/Fusion_Models/Full/model_best.pth.tar --classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```

* To validate a trained (on Fog Day) RGB-Gated Fusion (CBAM) model on Snow Night of STF Dataset, run the following command:
```
python validate_fusion.py Datasets/STF --dataset stf_fog_day --checkpoint Checkpoints/STF/Fusion_Models/All_Trained/Fog_Day/model_best.pth.tar --classwise --split test --num-classes 4 --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam
```

#### Validating Scene-Adaptive Models
```
# FLIR_Aligned dataset
bash bash/val_adaptive_flir.sh
# M3FD dataset
bash bash/val_adaptive_m3fd.sh
# STF dataset
bash bash/val_adaptive_stf.sh
# STF Clear dataset
bash bash/val_adaptive_stf_clear.sh
```

## Time benchmarks
To benchmark the scene-adaptive fusion model, create a new conda environment:
```
conda create -n dsfusion_benchmark python=3.9
conda activate dsfusion_benchmark
pip install -r benchmark_requirements.txt --no-deps
```
Then run:
```
bash bash/benchmark.sh
``` 
The benchmark assumes a batch size of `1` and a input image size of `768x768`. The data and model weights are mocked, so there is no need to copy data around when testing on multiple machines. The code will take around a minute to compile before conducting the inference benchmark. 
