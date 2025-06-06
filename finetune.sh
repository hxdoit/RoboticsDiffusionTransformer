#export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
#export NCCL_IB_DISABLE=0
#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFOpretrain
#export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-1b"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
#export CUTLASS_PATH="/path/to/cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

deepspeed  main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=4 \
    --sample_batch_size=2 \
    --max_train_steps=200000 \
    --checkpointing_period=400 \
    --sample_period=100 \
    --gradient_accumulation_steps=10 \
    --checkpoints_total_limit=2 \
    --num_sample_batches=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
