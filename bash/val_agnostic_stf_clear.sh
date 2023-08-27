python validate_fusion.py Datasets/STF --dataset stf_clear \
--checkpoint Checkpoints/STF/Fusion_Models/Clear_Trained/Clear/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam --classwise