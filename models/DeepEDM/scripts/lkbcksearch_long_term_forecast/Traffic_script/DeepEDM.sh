model_name=DeepEDM
prefix=$1
seed=$2
exp_idx=$3
prefix="Traffic_${prefix}_${seed}"
seq_len=512
if [ "$exp_idx" = "0" ] || [ "$exp_idx" = "" ]; then
  
  label_len=48
  pred_len=96

  python -u run.py \
      --task_name long_term_forecast \
      --model_config configs/benchmark_dataset.yaml \
      --is_training 1 \
      --condor_job True \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $prefix \
      --model DeepEDM \
      --seed $seed \
      --data custom \
      --features M \
      --loss mse_tdt\
      --n_mlp_layers 2 \
      --activation_fn selu \
      --n_edm_blocks 3 \
      --delay 19 \
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
      --label_len $label_len \
      --pred_len $pred_len \
      --itr 1
fi

if [ "$exp_idx" = "1" ] || [ "$exp_idx" = "" ]; then
  
  label_len=48
  pred_len=192

  python -u run.py \
      --task_name long_term_forecast \
      --model_config configs/benchmark_dataset.yaml \
      --is_training 1 \
      --condor_job True \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $prefix \
      --model DeepEDM \
      --seed $seed \
      --data custom \
      --features M \
      --loss mse_tdt\
      --n_mlp_layers 3 \
      --activation_fn selu \
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
      --label_len $label_len \
      --pred_len $pred_len \
      --itr 1 
fi

if [ "$exp_idx" = "2" ] || [ "$exp_idx" = "" ]; then
  
  label_len=48
  pred_len=336

  python -u run.py \
      --task_name long_term_forecast \
      --model_config configs/benchmark_dataset.yaml \
      --is_training 1 \
      --condor_job True \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $prefix \
      --model DeepEDM \
      --seed $seed \
      --data custom \
      --features M \
      --loss mse_tdt\
      --n_mlp_layers 2 \
      --activation_fn selu \
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
      --label_len $label_len \
      --pred_len $pred_len \
      --itr 1
fi

if [ "$exp_idx" = "3" ] || [ "$exp_idx" = "" ]; then
  seq_len=336
  label_len=48
  pred_len=720

  python -u run.py \
      --task_name long_term_forecast \
      --model_config configs/benchmark_dataset.yaml \
      --is_training 1 \
      --condor_job True \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $prefix \
      --model DeepEDM \
      --seed $seed \
      --data custom \
      --features M \
      --loss mse_tdt\
      --n_mlp_layers 3 \
      --activation_fn selu \
      --n_edm_blocks 2 \
      --delay 15 \
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
      --label_len $label_len \
      --pred_len $pred_len \
      --itr 1 
fi
