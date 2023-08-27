python validate_fusion_adaptive.py Datasets/FLIR_Aligned --dataset flir_aligned_full --num-scenes 3 \
--checkpoint Checkpoints/FLIR/Fusion_Models/Full/model_best.pth.tar \
--checkpoint-cls Checkpoints/FLIR/Classifier/flir_classifier.pth.tar \
--checkpoint-scenes Checkpoints/FLIR/Fusion_Models/Full/model_best.pth.tar \
Checkpoints/FLIR/Fusion_Models/Day/model_best.pth.tar \
Checkpoints/FLIR/Fusion_Models/Night/model_best.pth.tar \
--split test --num-classes 90 \
--rgb_mean 0.485 0.456 0.406 --rgb_std 0.229 0.224 0.225 --thermal_mean 0.519 0.519 0.519 --thermal_std 0.225 0.225 0.225 \
--classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam