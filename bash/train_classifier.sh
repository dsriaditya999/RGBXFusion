python train_classifier.py Datasets/M3FD --eval-metric=acc --dataset=m3fd_rgb --model=efficientdetv2_dt \
--batch-size=16 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=6 --num-scenes=5 \
--mean 0.49151019 0.50717567 0.50293698 --std 0.1623529 0.14178433 0.13799928 \
--workers=4 --initial-checkpoint Checkpoints/M3FD/Single_Modality_Models/m3fd_rgb_backbone.pth.tar