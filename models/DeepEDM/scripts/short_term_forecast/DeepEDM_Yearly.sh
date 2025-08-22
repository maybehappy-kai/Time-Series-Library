

model_name=DeepEDM
prefix=$1


#m4_Yearly
prefix="m4_Yearly_${prefix}"

python -u run.py \
    --task_name short_term_forecast \
    --condor_job True \
    --model_config configs/benchmark_dataset.yaml \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --loss SMAPE \
    --model_id $prefix \
    --model DeepEDM \
    --output_dir . \
    --data m4 \
    --features M \
    --activation_fn selu \
    --n_mlp_layers 2 \
    --n_edm_blocks 2 \
    --dist_projection_dim 32 \
    --delay 5 \
    --time_delay_stride 1 \
    --mlp_dropout 0.1 \
    --edm_dropout 0.1 \
    --learning_rate 0.005 \
    --reduce_lr_factor 0.99 \
    --train_epochs 250 \
    --batch_size 32 \
    --patience 40 \
    --min_lr 0.0005 \
    --itr 1