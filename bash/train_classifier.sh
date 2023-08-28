python train_classifier.py Datasets/FLIR_Aligned --eval-metric=acc --bench-task=train_cls --dataset=flir_aligned_rgb --model=efficientdetv2_dt \
    --batch-size=32 --amp --lr=1e-4 --opt adam --sched plateau --num-classes=90 --num-scenes=3\
    --mean 0.62721553 0.63597459 0.62891984 --std 0.16723704 0.17459581 0.18347738 \
    --save-images --workers=4 --initial-checkpoint /home/ganlu/train_flir_after_diffbb/rgb_checkpoint.pth.tar