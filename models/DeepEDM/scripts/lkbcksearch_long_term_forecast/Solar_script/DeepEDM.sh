model_name=DeepEDM
prefix=$1
seed=$2
exp_idx=$3
prefix="solar_${prefix}_${seed}"

seq_len=512

if [ "$exp_idx" = "0" ] || [ "$exp_idx" = "" ]; then
    pred_len=96
    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/solar \
        --data_path solar_AL.txt \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data Solar \
        --features M \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 7 \
        --time_delay_stride 1 \
        --mlp_dropout 0.2 \
        --edm_dropout 0.2 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 32 \
        --patience 10 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi

if [ "$exp_idx" = "1" ] || [ "$exp_idx" = "" ]; then
    pred_len=192
    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/solar \
        --data_path solar_AL.txt \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data Solar \
        --features M \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 7 \
        --time_delay_stride 1 \
        --mlp_dropout 0.2 \
        --edm_dropout 0.2 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 32 \
        --patience 10 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi

if [ "$exp_idx" = "2" ] || [ "$exp_idx" = "" ]; then
    pred_len=336
    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/solar \
        --data_path solar_AL.txt \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data Solar \
        --features M \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 7 \
        --time_delay_stride 1 \
        --mlp_dropout 0.2 \
        --edm_dropout 0.2 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 32 \
        --patience 10 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi

if [ "$exp_idx" = "3" ] || [ "$exp_idx" = "" ]; then
    pred_len=720
    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/solar \
        --data_path solar_AL.txt \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data Solar \
        --features M \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 7 \
        --time_delay_stride 1 \
        --mlp_dropout 0.2 \
        --edm_dropout 0.2 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 32 \
        --patience 10 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi
