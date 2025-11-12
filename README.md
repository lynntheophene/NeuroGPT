# NeuroGPT
### Neuro-GPT: Towards a Foundation Model for EEG  [paper](https://arxiv.org/abs/2311.03764)

#### Published on IEEE - ISBI 2024

We propose Neuro-GPT, a foundation model consisting of an EEG encoder and a GPT model. The foundation model is pre-trained on a large-scale data set using a self-supervised task that learns how to reconstruct masked EEG segments. We then fine-tune the model on a Motor Imagery Classification task to validate its performance in a low-data regime (9 subjects). Our experiments demonstrate that applying a foundation model can significantly improve classification performance compared to a model trained from scratch.
### Pre-trained foundation model available [here](https://huggingface.co/wenhuic/Neuro-GPT/tree/main)
<!-- 
<picture>
<source> -->
![Neuro-GPT Pipeline](./figures/pipeline.png)
<!-- </picture> -->
## Installation

### Standard Installation (GPU)
```console
git clone https://github.com/lynntheophene/NeuroGPT.git
cd NeuroGPT
pip install -r requirements.txt
cd scripts
./train.sh
```

### CPU-Only Installation (HP Pavilion & Laptops)
For systems without GPU (laptops, HP Pavilion, etc.):
```console
git clone https://github.com/lynntheophene/NeuroGPT.git
cd NeuroGPT
pip install -r requirements.txt
# Install CPU-only PyTorch (optional, if GPU version causes issues)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
cd scripts
./train_cpu.sh
```

**📖 See [CPU_TRAINING_GUIDE.md](CPU_TRAINING_GUIDE.md) for detailed CPU training instructions**

## Requirements
```console
pip install -r requirements.txt
```

For CPU-only systems, PyTorch will automatically use CPU. For optimal performance, you can install the CPU-specific version:
```console
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Datasets
- [TUH EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg)
- [BCI Competition IV 2a Dataset](https://www.bbci.de/competition/iv/#datasets)
- **[ZuCo-2 Dataset](https://osf.io/2urht/)** - EEG reading data (see [ZUCO2_SETUP.md](ZUCO2_SETUP.md))

## Quick Start

### GPU Training
```bash
cd scripts
./train.sh  # Standard GPU training
```

### CPU Training (HP Pavilion & Laptops)
```bash
cd scripts
./train_cpu.sh  # CPU-optimized training
```

### ZuCo-2 Dataset Training
```bash
# 1. Download and preprocess ZuCo-2 (see ZUCO2_SETUP.md)
cd src/batcher
python3 zuco_dataset.py --input /path/to/zuco2 --output /path/to/zuco2_preprocessed

# 2. Train on CPU
cd ../../scripts
./finetune_cpu_zuco2.sh /path/to/zuco2_preprocessed
```

**📖 Full ZuCo-2 setup guide: [ZUCO2_SETUP.md](ZUCO2_SETUP.md)**

## Documentation

- **[CPU_TRAINING_GUIDE.md](CPU_TRAINING_GUIDE.md)** - Complete guide for running on CPU (HP Pavilion, laptops)
- **[ZUCO2_SETUP.md](ZUCO2_SETUP.md)** - ZuCo-2 dataset download, preprocessing, and training
- **[Original Paper](https://arxiv.org/abs/2311.03764)** - NeuroGPT research paper

## Acknowledgments
This project is developed based on the following open-source repositories:
- [Self-supervised learning of brain dynamics from broad neuroimaging data](https://github.com/athms/learning-from-brains)
- [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
