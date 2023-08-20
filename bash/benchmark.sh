python time_benchmark.py /home/carson/data/FLIR/FLIR_Aligned \
--dataset flir_aligned_full \
--num-scenes 3 \
--split test \
--num-classes 90 \
--model efficientdetv2_dt \
--branch fusion \
--att_type cbam