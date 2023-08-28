modality=$1

python train_single.py \
Datasets/FLIR_Aligned \
--dataset=flir_aligned_${modality} \
--model=efficientdetv2_dt \
--batch-size=32 \
--amp \
--lr=1e-3 \
--opt adam \
--sched plateau \
--num-classes=3 \
--save-images \
--workers=8 \
--pretrained \
--mean 0.53584253 0.53584253 0.53584253 \
--std 0.24790472 0.24790472 0.24790472