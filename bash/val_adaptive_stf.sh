python validate_fusion_adaptive.py Datasets/STF --dataset stf_full --num-scenes 7 \
--checkpoint Checkpoints/STF/Fusion_Models/All_Trained/Full/model_best.pth \
--checkpoint-cls Checkpoints/STF/Classifiers/stf_classifier_all_trained.pth.tar \
--checkpoint-scenes Checkpoints/STF/Fusion_Models/All_Trained/Full/model_best.pth \
Checkpoints/STF/Fusion_Models/All_Trained/Clear_Day/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/All_Trained/Clear_Night/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/All_Trained/Fog_Day/model_best.pth \
Checkpoints/STF/Fusion_Models/All_Trained/Fog_Night/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/All_Trained/Snow_Day/model_best.pth.tar \
Checkpoints/STF/Fusion_Models/All_Trained/Snow_Night/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam