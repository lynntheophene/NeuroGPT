#!/bin/bash
#
# CPU-Optimized Fine-tuning Script for ZuCo-2 Dataset
#
# This script fine-tunes a pretrained NeuroGPT model on ZuCo-2 dataset
# using CPU-friendly settings for HP Pavilion and similar consumer laptops.
#
# Prerequisites:
# - Preprocessed ZuCo-2 data (see ZUCO2_SETUP.md)
# - Pretrained model checkpoint (optional but recommended)
#

echo "Starting CPU-optimized NeuroGPT fine-tuning on ZuCo-2..."
echo "This configuration is designed for HP Pavilion and similar consumer laptops"
echo ""

# Navigate to the source directory
cd "$(dirname "$0")/../src" || exit 1

# Check if data path is provided
if [ -z "$1" ]; then
    echo "Usage: ./finetune_cpu_zuco2.sh <path_to_zuco2_data> [optional_pretrained_model_path]"
    echo ""
    echo "Example:"
    echo "  ./finetune_cpu_zuco2.sh ../../zuco2_preprocessed/"
    echo "  ./finetune_cpu_zuco2.sh ../../zuco2_preprocessed/ ../pretrained_model/pytorch_model.bin"
    exit 1
fi

ZUCO2_DATA_PATH="$1"
PRETRAINED_MODEL="${2:-none}"

echo "ZuCo-2 Data Path: $ZUCO2_DATA_PATH"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo ""

# CPU-friendly hyperparameters for fine-tuning:
# - Very small batch size (2) to minimize memory usage
# - Fewer training steps
# - Disabled fp16 (CPU doesn't support it)
# - Small model configuration
# - Single worker to avoid multiprocessing overhead

python3 train_gpt.py \
    --training-style='CSM_causal' \
    --training-steps=2000 \
    --eval_every_n_steps=200 \
    --log-every-n-steps=200 \
    --per-device-training-batch-size=2 \
    --per-device-validation-batch-size=2 \
    --num-workers=0 \
    --num_chunks=4 \
    --chunk_len=500 \
    --chunk_ovlp=50 \
    --num-hidden-layers=2 \
    --num-encoder-layers=2 \
    --embedding-dim=256 \
    --learning-rate=5e-5 \
    --fp16='False' \
    --deepspeed='none' \
    --use-encoder='True' \
    --do-normalization='True' \
    --run-name='zuco2_cpu_finetune' \
    --train-data-path="$ZUCO2_DATA_PATH" \
    --pretrained-model="$PRETRAINED_MODEL"

echo ""
echo "Fine-tuning complete! Check the results/ directory for outputs."
