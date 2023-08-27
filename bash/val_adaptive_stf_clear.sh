python validate_fusion_adaptive.py Datasets/STF --dataset stf_clear --num-scenes 3 \
--checkpoint Checkpoints/STF/Fusion_Models/Clear_Trained/Clear/model_best.pth.tar \
--checkpoint-cls Checkpoints/STF/Classifiers/stf_classifier_clear_trained.pth.tar \
--checkpoint-scenes Checkpoints/STF/Fusion_Models/Clear_Trained/Clear/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/Clear_Trained/Clear_Day/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/Clear_Trained/Clear_Night/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam --classwise