python validate_fusion.py Datasets/M3FD --dataset m3fd_full \
--checkpoint Checkpoints/M3FD/Fusion_Models/Full/model_best.pth.tar \
--split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam