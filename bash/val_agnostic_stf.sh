python validate_fusion.py Datasets/STF --dataset stf_full \
--checkpoint Checkpoints/STF/Fusion_Models/All_Trained/Full/model_best.pth \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam