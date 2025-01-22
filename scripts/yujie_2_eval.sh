model_id="meta-llama/Llama-3.2-1B"
save_folder=logs/$model_id-300steps

torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    ptq.py \
    --input_model $model_id \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --w_bits 4 \
    --a_bits 4 \
    --k_bits 16 \
    --v_bits 16 \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --rotate \
    --optimized_rotation_path "$save_folder/rotation/R.bin"
