# Quick Start - Running NeuroGPT on HP Pavilion (CPU)

This is a condensed quick-start guide for running NeuroGPT on CPU-only systems like HP Pavilion laptops.

## 1. Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/lynntheophene/NeuroGPT.git
cd NeuroGPT

# Install dependencies
pip install -r requirements.txt

# Optional: Install CPU-only PyTorch (if you have GPU version installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Check your system
python3 check_system.py
```

## 2. Verify Installation (1 minute)

```bash
python3 example_cpu_training.py
```

This shows available configurations and example commands.

## 3. Quick Test (5 minutes)

Run a smoke test to verify everything works:

```bash
cd src
python3 train_gpt.py \
    --smoke-test='True' \
    --fp16='False' \
    --deepspeed='none' \
    --num-workers=0
```

## 4. Training Options

### Option A: CPU Training Script (Easiest)
```bash
cd scripts
./train_cpu.sh
```

### Option B: ZuCo-2 Dataset

#### Download ZuCo-2
1. Visit https://osf.io/2urht/
2. Create free OSF account
3. Download dataset files
4. Save to `~/datasets/zuco2/`

#### Preprocess Data
```bash
cd src/batcher
python3 zuco_dataset.py \
    --input ~/datasets/zuco2/ \
    --output ~/datasets/zuco2_preprocessed/
```

#### Train on ZuCo-2
```bash
cd ../../scripts
./finetune_cpu_zuco2.sh ~/datasets/zuco2_preprocessed/
```

## 5. Configuration Presets

For different RAM sizes:

```bash
cd src

# 8GB RAM (minimal)
python3 -c "from cpu_config import get_cpu_config; print(get_cpu_config('minimal'))"

# 16GB RAM (standard - recommended for HP Pavilion)
python3 -c "from cpu_config import get_cpu_config; print(get_cpu_config('standard'))"

# 32GB+ RAM (high-end)
python3 -c "from cpu_config import get_cpu_config; print(get_cpu_config('high_end'))"
```

## 6. Expected Performance (HP Pavilion, Core i5, 16GB RAM)

| Configuration | Training Speed | Total Time (5000 steps) |
|--------------|----------------|-------------------------|
| Minimal (8GB) | ~5-8 sec/step | ~7-11 hours |
| Standard (16GB) | ~2-4 sec/step | ~3-6 hours |
| Quick Test | ~2-3 sec/step | ~5 minutes |

## 7. Common Issues

### Out of Memory
- Reduce batch size: `--per-device-training-batch-size=1`
- Reduce model size: `--embedding-dim=128 --num-hidden-layers=2`
- Close other applications

### Training Too Slow
- Use fewer steps: `--training-steps=1000`
- Use smaller model
- Be patient - CPU training is 10-50x slower than GPU

### "fp16 requires CUDA" Error
- Make sure you have `--fp16='False'` in your command

## 8. Full Documentation

- **[CPU_TRAINING_GUIDE.md](CPU_TRAINING_GUIDE.md)** - Complete CPU training guide
- **[ZUCO2_SETUP.md](ZUCO2_SETUP.md)** - ZuCo-2 dataset guide
- **[README.md](README.md)** - Main project README

## 9. Get Help

```bash
python3 check_system.py        # Check system requirements
python3 example_cpu_training.py # Show quick start info
python3 src/train_gpt.py --help # Show all training options
```

## Summary

1. **Install**: Clone repo, install dependencies
2. **Verify**: Run check_system.py and example_cpu_training.py
3. **Test**: Run smoke test (5 minutes)
4. **Train**: Use train_cpu.sh or configure manually
5. **Monitor**: Check results/ directory for outputs

**For HP Pavilion (16GB RAM)**: Use the `standard` configuration preset or run `./train_cpu.sh`

**For ZuCo-2 Dataset**: Follow the 3-step process: Download → Preprocess → Train
