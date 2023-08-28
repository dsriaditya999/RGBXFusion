############################################# FLIR Aligned #################################################
python validate_classifier.py Datasets/FLIR_Aligned --dataset=flir_aligned_rgb --model efficientdetv2_dt --workers 8 \
--checkpoint Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --num-classes=3 --num-scenes=3 \
-b 8 --split test

############################################# M3FD #################################################
# python validate_classifier.py Datasets/M3FD --dataset=m3fd_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint Checkpoints/M3FD/Classifier/m3fd_classifier.pth.tar --num-classes=6 --num-scenes=5 \
# -b 8 --split test

############################################# STF Clear #################################################
# python validate_classifier.py Datasets/STF --dataset=stf_clear_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint Checkpoints/STF/Classifiers/stf_classifier_clear_trained.pth.tar --num-classes=4 --num-scenes=3 \
# -b 12 --split test --img-size 1280

############################################# STF #################################################
# python validate_classifier.py Datasets/STF --dataset=stf_full_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint Checkpoints/STF/Classifiers/stf_classifier_all_trained.pth.tar --num-classes=4 --num-scenes=7 \
# -b 12 --split test --img-size 1280