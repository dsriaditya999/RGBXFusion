# python validate_classifier.py Datasets/FLIR_Aligned --dataset=flir_aligned_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint Checkpoints/FLIR_Aligned/Classifier/flir_classifier.pth.tar --num-classes=3 --num-scenes=3 \
# -b 8 --split test

# python validate_classifier.py /home/carson/data/m3fd --dataset=m3fd_rgb --model efficientdetv2_dt --workers 8 \
# --checkpoint Checkpoints/M3FD/Classifier/m3fd_classifier.pth.tar --num-classes=6 --num-scenes=5 \
# -b 8 --split test

python validate_classifier.py /media/hdd2/rgb_gated_aligned/ --dataset=stf_clear_rgb --model efficientdetv2_dt --workers 8 \
--checkpoint /home/ganlu/workspace/efficientdet-pytorch/output/train/stf-rgb-backbone-cls/model_best.pth.tar --num-classes=4 --num-scenes=3 \
-b 12 --split test --img-size 1280