time=$(date "+%Y%m%d-%H%M%S")

CUDA_VISIBLE_DEVICES=0 /home/gbc/.conda/envs/rookie/bin/python \
train.py \
--space nasbench \
--data data/nas_bench.json \
--train_batch_size 256 \
--output output/runtime_exp_seed22222222_${time} \
--trainset_size 4236 --valset_size 1000 --seed 22222222
