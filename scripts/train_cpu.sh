#!/bin/bash
#
# CPU-Optimized Training Script for NeuroGPT on HP Pavilion or similar laptops
#
# This script is configured to run on CPU with reduced memory requirements
# suitable for consumer-grade hardware like HP Pavilion laptops.
#
# Memory Requirements: ~8-16GB RAM
# Training Time: Slower than GPU but feasible for small datasets
#

echo "Starting CPU-optimized NeuroGPT training..."
echo "This configuration is designed for HP Pavilion and similar consumer laptops"
echo ""

# Navigate to the source directory
cd "$(dirname "$0")/../src" || exit 1

# CPU-friendly hyperparameters:
# - Smaller batch size to reduce memory usage
# - Fewer training steps for faster experimentation
# - Disabled fp16 (requires GPU)
# - Disabled deepspeed (requires GPU)
# - Reduced model size with fewer layers and smaller embedding dimension
# - More workers set to 0 to avoid multiprocessing overhead on CPU

python3 train_gpt.py \
    --training-style='CSM_causal' \
    --training-steps=5000 \
    --eval_every_n_steps=500 \
    --log-every-n-steps=500 \
    --per-device-training-batch-size=4 \
    --per-device-validation-batch-size=4 \
    --num-workers=0 \
    --num_chunks=8 \
    --chunk_len=500 \
    --chunk_ovlp=50 \
    --num-hidden-layers=2 \
    --num-encoder-layers=2 \
    --embedding-dim=256 \
    --learning-rate=1e-4 \
    --fp16='False' \
    --deepspeed='none' \
    --run-name='cpu_training' \
    --train-data-path='../../tuh_tensors/' \
    "$@"

echo ""
echo "Training complete! Check the results/ directory for outputs."
