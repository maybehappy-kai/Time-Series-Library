

model_name=DeepEDM
prefix=$1
seed=$2
exp_idx=$3
prefix="ECL_${prefix}_${seed}"

seq_lens=(96 192 288 384)
pred_lens=(48 96 144 192)


if [ "$exp_idx" = "0" ] || [ "$exp_idx" = "" ]; then
    seq_len=96
    pred_len=48
    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/electricity \
        --data_path electricity.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --activation_fn selu \
        --loss mse_tdt \
        --n_mlp_layers 2 \
        --n_edm_blocks 3 \
        --delay 21 \
        --time_delay_stride 2 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.001 \
        --reduce_lr_factor 0.99 \
        --train_epochs 250 \
        --batch_size 32 \
        --patience 15 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1

fi

if [ "$exp_idx" = "1" ] || [ "$exp_idx" = "" ]; then
    seq_len=192
    pred_len=96

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/electricity \
        --data_path electricity.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --activation_fn selu \
        --loss mse_tdt \
        --n_mlp_layers 3 \
        --n_edm_blocks 2 \
        --delay 21 \
        --time_delay_stride 2 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.001 \
        --reduce_lr_factor 0.99 \
        --train_epochs 250 \
        --batch_size 32 \
        --patience 15 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi



if [ "$exp_idx" = "2" ] || [ "$exp_idx" = "" ]; then
    seq_len=288
    pred_len=144

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/electricity \
        --data_path electricity.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --activation_fn selu \
        --loss mse_tdt \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 19 \
        --time_delay_stride 4 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.001 \
        --reduce_lr_factor 0.99 \
        --train_epochs 250 \
        --batch_size 32 \
        --patience 15 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi


if [ "$exp_idx" = "3" ] || [ "$exp_idx" = "" ]; then
    seq_len=384
    pred_len=192

    python -u run.py \
        --task_name long_term_forecast \
        --model_config configs/benchmark_dataset.yaml \
        --is_training 1 \
        --root_path ./dataset/electricity \
        --data_path electricity.csv \
        --model_id $prefix \
        --model DeepEDM \
        --seed $seed \
        --output_dir . \
        --data custom \
        --features M \
        --activation_fn selu \
        --loss mse_tdt \
        --n_mlp_layers 2 \
        --n_edm_blocks 2 \
        --delay 19 \
        --time_delay_stride 4 \
        --mlp_dropout 0.1 \
        --edm_dropout 0.1 \
        --learning_rate 0.001 \
        --reduce_lr_factor 0.99 \
        --train_epochs 250 \
        --batch_size 32 \
        --patience 15 \
        --min_lr 0.00005 \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --itr 1
fi