model_id="meta-llama/Llama-3.2-1B"
save_folder=logs/$model_id-300steps
mkdir -p $save_folder/rotation
mkdir -p $save_folder/logging
mkdir -p $save_folder/output

torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d -rdzv-endpoint=localhost:0 \
    optimize_rotation.py \
    --input_model $model_id \
    --output_rotation_path "$save_folder/rotation/" \
    --output_dir "$save_folder/output/" \
    --logging_dir "$save_folder/logging/" \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --per_device_train_batch_size 4 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --save_safetensors False \
    --max_steps 300 \
    --w_bits 4 \
    --a_bits 4 \
    --k_bits 16 \
    --v_bits 16 \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --k_groupsize 128 \
    --v_groupsize 128
