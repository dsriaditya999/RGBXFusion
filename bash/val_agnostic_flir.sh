python validate_fusion.py Datasets/FLIR_Aligned --dataset flir_aligned_full \
--checkpoint Checkpoints/FLIR_Aligned/Fusion_Models/Full/model_best.pth.tar \
--split test --num-classes 90 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam --classwise