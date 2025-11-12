# Running NeuroGPT on CPU (HP Pavilion & Similar Laptops)

This guide explains how to run NeuroGPT on CPU-only systems like HP Pavilion laptops without requiring a GPU.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Training Options](#training-options)
5. [ZuCo-2 Dataset Setup](#zuco-2-dataset-setup)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or AMD Ryzen 5 (or equivalent)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space (more for datasets)
- **OS**: Windows 10/11, Linux, or macOS

### Recommended for HP Pavilion
- **CPU**: Intel Core i5-11th gen or newer / AMD Ryzen 5 5000 series or newer
- **RAM**: 16GB
- **Storage**: SSD with 20GB+ free space

### Software Requirements
- Python 3.8 or newer
- pip package manager

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/lynntheophene/NeuroGPT.git
cd NeuroGPT
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for CPU-only systems**: The default PyTorch installation should work fine. If you encounter issues, install the CPU-only version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

This should show PyTorch version and `CUDA available: False` for CPU-only setups.

## Quick Start

### Option 1: Training from Scratch (Small Dataset)

For quick experimentation or when you have limited data:

```bash
cd scripts
chmod +x train_cpu.sh
./train_cpu.sh
```

This will start training with CPU-optimized settings:
- Small model size (2 layers, 256 embedding dimension)
- Small batch size (4 samples)
- 5000 training steps (~30-60 minutes on modern CPUs)

### Option 2: Smoke Test (Fastest)

To verify everything works without waiting for full training:

```bash
cd src
python3 train_gpt.py \
    --smoke-test='True' \
    --fp16='False' \
    --deepspeed='none' \
    --num-workers=0 \
    --train-data-path='../../tuh_tensors/'
```

This runs a minimal training loop and completes in 1-2 minutes.

## Training Options

### CPU-Optimized Training Script

The `train_cpu.sh` script is pre-configured for CPU training. You can customize it:

```bash
cd scripts
./train_cpu.sh \
    --training-steps=10000 \
    --per-device-training-batch-size=2 \
    --embedding-dim=128
```

### Key Parameters for CPU Training

| Parameter | CPU-Friendly Value | Description |
|-----------|-------------------|-------------|
| `--fp16` | `'False'` | **Must be False** - FP16 requires GPU |
| `--deepspeed` | `'none'` | **Must be none** - Deepspeed requires GPU |
| `--num-workers` | `0` | Avoid multiprocessing overhead on CPU |
| `--per-device-training-batch-size` | `2-4` | Small batch to reduce memory usage |
| `--embedding-dim` | `128-256` | Smaller dimension = less memory |
| `--num-hidden-layers` | `2-4` | Fewer layers = faster training |
| `--num-encoder-layers` | `2-4` | Fewer encoder layers |
| `--training-steps` | `2000-5000` | Fewer steps for faster experimentation |

### Manual Training Command

For full control:

```bash
cd src
python3 train_gpt.py \
    --training-style='CSM_causal' \
    --training-steps=5000 \
    --per-device-training-batch-size=4 \
    --per-device-validation-batch-size=4 \
    --num-workers=0 \
    --num_chunks=8 \
    --chunk_len=500 \
    --num-hidden-layers=2 \
    --num-encoder-layers=2 \
    --embedding-dim=256 \
    --fp16='False' \
    --deepspeed='none' \
    --train-data-path='../../tuh_tensors/'
```

## ZuCo-2 Dataset Setup

The ZuCo-2 (Zurich Cognitive Language Processing Corpus) dataset contains EEG data recorded while subjects read text.

### Step 1: Download ZuCo-2 Dataset

1. Visit the ZuCo-2 dataset page: https://osf.io/2urht/
2. Download the dataset files (requires free OSF account)
3. Extract the files to a directory, e.g., `/path/to/zuco2/`

The ZuCo-2 dataset contains:
- Task 1: Normal reading
- Task 2: Task-specific reading
- Task 3: Reading and relation extraction

### Step 2: Preprocess ZuCo-2 Data

ZuCo-2 data needs to be converted to the format expected by NeuroGPT:

```bash
cd src/batcher
python3 zuco_dataset.py \
    --input /path/to/zuco2/ \
    --output /path/to/zuco2_preprocessed/
```

This will:
- Load ZuCo-2 .mat files
- Map the 105-channel EEG data to the standard 22-channel configuration
- Save as PyTorch tensors for efficient loading

### Step 3: Train on ZuCo-2

#### Option A: Using the provided script (Recommended)

```bash
cd scripts
chmod +x finetune_cpu_zuco2.sh
./finetune_cpu_zuco2.sh /path/to/zuco2_preprocessed/
```

With a pretrained model:

```bash
./finetune_cpu_zuco2.sh /path/to/zuco2_preprocessed/ ../pretrained_model/pytorch_model.bin
```

#### Option B: Manual command

```bash
cd src
python3 train_gpt.py \
    --training-style='CSM_causal' \
    --training-steps=2000 \
    --per-device-training-batch-size=2 \
    --num-workers=0 \
    --num_chunks=4 \
    --chunk_len=500 \
    --num-hidden-layers=2 \
    --embedding-dim=256 \
    --fp16='False' \
    --deepspeed='none' \
    --train-data-path='/path/to/zuco2_preprocessed/' \
    --run-name='zuco2_training'
```

### Expected Training Time on HP Pavilion

Approximate times for ZuCo-2 training on a typical HP Pavilion (Core i5, 16GB RAM):

| Configuration | Time per Step | Total Time (2000 steps) |
|--------------|---------------|------------------------|
| Batch=2, Layers=2, Embed=128 | ~1-2 seconds | ~1-1.5 hours |
| Batch=2, Layers=2, Embed=256 | ~2-3 seconds | ~1.5-2 hours |
| Batch=4, Layers=4, Embed=512 | ~5-7 seconds | ~3-4 hours |

**Tip**: Start with smaller configurations and increase gradually based on your available memory.

## Troubleshooting

### Issue: Out of Memory Error

**Solutions**:
1. Reduce batch size:
   ```bash
   --per-device-training-batch-size=1
   ```

2. Reduce model size:
   ```bash
   --embedding-dim=128 --num-hidden-layers=2
   ```

3. Reduce number of chunks:
   ```bash
   --num_chunks=4
   ```

4. Close other applications to free up RAM

### Issue: Training is Very Slow

**Expected behavior**: CPU training is 10-50x slower than GPU training.

**Optimization tips**:
1. Use fewer training steps for experimentation:
   ```bash
   --training-steps=1000
   ```

2. Reduce validation frequency:
   ```bash
   --eval_every_n_steps=1000
   ```

3. Use a smaller model:
   ```bash
   --num-hidden-layers=2 --embedding-dim=128
   ```

### Issue: "CUDA out of memory" Error on CPU System

This error shouldn't occur on CPU. If it does:

1. Verify PyTorch is using CPU:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be False
   ```

2. Reinstall CPU-only PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

### Issue: "fp16 requires CUDA" Error

**Solution**: Make sure you set `--fp16='False'` in your training command.

### Issue: Multiprocessing Errors

**Solution**: Set `--num-workers=0` to disable multiprocessing:
```bash
--num-workers=0
```

### Issue: Cannot Find Dataset

**Solutions**:
1. Check that your data path is correct and accessible
2. For ZuCo-2, ensure you ran the preprocessing step
3. Use absolute paths instead of relative paths:
   ```bash
   --train-data-path='/absolute/path/to/data/'
   ```

### Issue: Python Module Not Found

**Solution**: Make sure you're in the correct directory and Python can find the modules:

```bash
cd NeuroGPT/src
export PYTHONPATH="${PYTHONPATH}:/path/to/NeuroGPT/src"
python3 train_gpt.py ...
```

## Performance Tips

### 1. Use SSD Storage
Store your dataset on an SSD rather than HDD for faster data loading.

### 2. Monitor System Resources
Use task manager (Windows) or `htop` (Linux) to monitor RAM and CPU usage:
```bash
# Linux/Mac
htop
```

### 3. Adjust Based on Your Hardware

For systems with **8GB RAM**:
```bash
--per-device-training-batch-size=1 \
--embedding-dim=128 \
--num-hidden-layers=2 \
--num_chunks=4
```

For systems with **16GB+ RAM**:
```bash
--per-device-training-batch-size=4 \
--embedding-dim=256 \
--num-hidden-layers=4 \
--num_chunks=8
```

### 4. Background Processes
Close unnecessary applications and browser tabs to free up RAM for training.

### 5. Cooling
Ensure proper laptop cooling (use on hard surface, consider cooling pad) as CPU training can be intensive.

## Next Steps

1. **Start Small**: Begin with the smoke test, then try `train_cpu.sh`
2. **Monitor Progress**: Check the `results/` directory for logs and checkpoints
3. **Experiment**: Adjust hyperparameters based on your hardware
4. **Scale Up**: Once comfortable, try larger models or longer training

## Additional Resources

- [Original NeuroGPT Paper](https://arxiv.org/abs/2311.03764)
- [ZuCo-2 Dataset](https://osf.io/2urht/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Support

If you encounter issues not covered here:
1. Check the main README.md
2. Review the error messages carefully
3. Try the smoke test to isolate the problem
4. Open an issue on GitHub with system details and error logs

---

**Note**: This guide is optimized for consumer-grade laptops. While training will be slower than GPU, it enables running NeuroGPT on accessible hardware.
