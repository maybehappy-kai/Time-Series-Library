

model_name=DeepEDM
prefix=$1
seed=$2
exp_idx=$3
prefix="ili_${prefix}_${seed}"


if [ "$exp_idx" = "0" ] || [ "$exp_idx" = "" ]; then
    seq_len=48
    pred_len=24

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/illness \
        --data_path national_illness.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --n_mlp_layers 3 \
        --channel_mixer False \
        --n_edm_blocks 3 \
        --delay 9 \
        --time_delay_stride 1 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 8 \
        --patience 45 \
        --min_lr 0.0002 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi

if [ "$exp_idx" = "1" ] || [ "$exp_idx" = "" ]; then
    seq_len=72
    pred_len=36

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/illness \
        --data_path national_illness.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --n_mlp_layers 3 \
        --channel_mixer False \
        --n_edm_blocks 3 \
        --delay 9 \
        --time_delay_stride 1 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 8 \
        --patience 45 \
        --min_lr 0.0002 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi


if [ "$exp_idx" = "2" ] || [ "$exp_idx" = "" ]; then
    seq_len=96
    pred_len=48

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/illness \
        --data_path national_illness.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --n_mlp_layers 3 \
        --channel_mixer False \
        --n_edm_blocks 3 \
        --delay 9 \
        --time_delay_stride 1 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 8 \
        --patience 45 \
        --min_lr 0.0002 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi

if [ "$exp_idx" = "3" ] || [ "$exp_idx" = "" ]; then
    seq_len=120
    pred_len=60

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/illness \
        --data_path national_illness.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --n_mlp_layers 3 \
        --channel_mixer False \
        --n_edm_blocks 3 \
        --delay 9 \
        --time_delay_stride 1 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.0005 \
        --reduce_lr_factor 0.9 \
        --train_epochs 150 \
        --batch_size 8 \
        --patience 45 \
        --min_lr 0.0002 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi