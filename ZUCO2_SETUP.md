# ZuCo-2 Dataset Setup Guide for NeuroGPT

This guide provides detailed instructions for downloading, preprocessing, and using the ZuCo-2 (Zurich Cognitive Language Processing Corpus) dataset with NeuroGPT on CPU.

## About ZuCo-2

**ZuCo-2** is an EEG dataset containing brain activity recordings from subjects reading text. It's particularly useful for:
- Natural language processing with neural signals
- Reading comprehension studies
- EEG-based language modeling

**Citation**: If you use ZuCo-2, please cite:
```
Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). 
ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. 
Scientific Data, 5, 180291.
```

### Dataset Characteristics

- **Subjects**: 18 participants
- **Recording Type**: EEG (electroencephalography)
- **Channels**: 105 EEG channels (10-20 system extended)
- **Sampling Rate**: 500 Hz
- **Tasks**: 
  - Task 1: Normal reading (NR)
  - Task 2: Task-specific reading (TSR)
  - Task 3: Reading and relation extraction (RR)
- **Size**: ~4-5 GB (compressed)

## Step 1: Download ZuCo-2 Dataset

### Option A: Direct Download from OSF

1. Visit the official ZuCo-2 repository: https://osf.io/2urht/

2. You will need to create a free OSF (Open Science Framework) account if you don't have one

3. Download the dataset files:
   - Navigate to the "Files" section
   - Download all `.mat` files for the tasks you want to use
   - Recommended: Download Task 1 (Normal Reading) first for initial experimentation

4. Create a directory structure:
   ```bash
   mkdir -p ~/datasets/zuco2
   mkdir -p ~/datasets/zuco2/task1-NR
   mkdir -p ~/datasets/zuco2/task2-TSR
   mkdir -p ~/datasets/zuco2/task3-RR
   ```

5. Extract the downloaded files to the appropriate directories

### Option B: Using wget (if direct links are available)

```bash
# Create directory
mkdir -p ~/datasets/zuco2/raw
cd ~/datasets/zuco2/raw

# Download files (example - check OSF for actual URLs)
# Note: You may need to authenticate
wget <download_url_from_osf>
```

### Verify Download

Check that you have the expected files:

```bash
cd ~/datasets/zuco2
find . -name "*.mat" | wc -l
# Should show multiple .mat files (one per subject per task)
```

## Step 2: Understanding ZuCo-2 File Structure

Each `.mat` file contains:

```
- rawData: Raw EEG signals [channels × time_points]
- sentenceData: Sentence-level data structure
  - content: Text of the sentence
  - word: Word-level information
  - rawData: EEG data for this sentence
- channelLabels: Names of EEG channels
- sampling_rate: 500 Hz
```

### Example: Inspecting a File

```python
import scipy.io as sio
import numpy as np

# Load a sample file
mat_data = sio.loadmat('path/to/zuco2/task1/ZDM_NR.mat')

# Check available keys
print("Keys:", mat_data.keys())

# Check data shape
print("Raw data shape:", mat_data['rawData'].shape)

# Check channel labels
print("Channels:", mat_data['channelLabels'])
```

## Step 3: Preprocess ZuCo-2 for NeuroGPT

NeuroGPT expects data in a specific format with 22 standard EEG channels. We need to preprocess ZuCo-2's 105 channels.

### Automated Preprocessing

Use the provided preprocessing script:

```bash
cd NeuroGPT/src/batcher

# Preprocess all tasks
python3 zuco_dataset.py \
    --input ~/datasets/zuco2/raw/ \
    --output ~/datasets/zuco2_preprocessed/

# Or preprocess a specific task
python3 zuco_dataset.py \
    --input ~/datasets/zuco2/task1-NR/ \
    --output ~/datasets/zuco2_preprocessed/task1/
```

This script will:
1. Load each `.mat` file
2. Extract the EEG data
3. Map 105 channels → 22 standard channels
4. Normalize the data
5. Save as PyTorch tensors (`.pt` files)

### Manual Preprocessing (Advanced)

If you need custom preprocessing:

```python
from src.batcher.zuco_dataset import ZuCo2Dataset, prepare_zuco2_data
import torch
import numpy as np

# Custom preprocessing
def custom_preprocess(mat_file_path, output_path):
    # Load data
    dataset = ZuCo2Dataset(
        filenames=[mat_file_path],
        sample_keys=['inputs', 'attention_mask'],
        chunk_len=500,
        num_chunks=10,
        ovlp=50
    )
    
    # Process
    eeg_data = dataset.load_zuco_mat_file(mat_file_path)
    
    # Additional custom processing here
    # e.g., filtering, artifact removal, etc.
    
    # Save
    torch.save(torch.from_numpy(eeg_data), output_path)

# Use it
custom_preprocess(
    'path/to/ZDM_NR.mat',
    'path/to/output/ZDM_NR.pt'
)
```

### Verify Preprocessing

Check the preprocessed data:

```bash
cd ~/datasets/zuco2_preprocessed
ls -lh
# Should show .pt files corresponding to your .mat files

# Check a file
python3 -c "
import torch
data = torch.load('ZDM_NR.pt')
print(f'Shape: {data.shape}')
print(f'Data type: {data.dtype}')
print(f'Min/Max: {data.min():.3f} / {data.max():.3f}')
"
```

Expected output:
```
Shape: torch.Size([22, N])  # 22 channels, N time points
Data type: torch.float32
Min/Max: -X.XXX / X.XXX
```

## Step 4: Organize Data for Training

### Create Subject Lists

For better control over train/validation splits:

```bash
cd ~/datasets/zuco2_preprocessed

# List all processed files
ls *.pt > subject_list.txt

# Or use Python to create a filtered list
python3 << EOF
import os
files = [f for f in os.listdir('.') if f.endswith('.pt')]
# Remove any files shorter than threshold if needed
with open('subject_list.txt', 'w') as f:
    for file in files:
        f.write(file + '\n')
print(f"Created list with {len(files)} subjects")
EOF
```

### Split Train/Validation

```python
# split_dataset.py
import os
import random

def split_train_val(file_list, train_ratio=0.8):
    """Split files into train and validation sets"""
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    with open('train_files.txt', 'w') as f:
        f.writelines([f + '\n' for f in train_files])
    
    with open('val_files.txt', 'w') as f:
        f.writelines([f + '\n' for f in val_files])
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")

if __name__ == '__main__':
    split_train_val('subject_list.txt')
```

Run it:
```bash
python3 split_dataset.py
```

## Step 5: Training on ZuCo-2

### Quick Start Training

Use the provided CPU-optimized script:

```bash
cd NeuroGPT/scripts
chmod +x finetune_cpu_zuco2.sh

# Basic training
./finetune_cpu_zuco2.sh ~/datasets/zuco2_preprocessed/
```

### Training with Pretrained Model

For better results, use a pretrained model:

```bash
# Download pretrained model (if available)
mkdir -p ~/models/neurogpt_pretrained
# Download from HuggingFace or the paper repository

# Train with pretrained model
./finetune_cpu_zuco2.sh \
    ~/datasets/zuco2_preprocessed/ \
    ~/models/neurogpt_pretrained/pytorch_model.bin
```

### Custom Training Configuration

For full control:

```bash
cd NeuroGPT/src

python3 train_gpt.py \
    --training-style='CSM_causal' \
    --train-data-path=~/datasets/zuco2_preprocessed/ \
    --training-steps=3000 \
    --eval_every_n_steps=300 \
    --log-every-n-steps=300 \
    --per-device-training-batch-size=2 \
    --per-device-validation-batch-size=2 \
    --num-workers=0 \
    --num_chunks=8 \
    --chunk_len=500 \
    --chunk_ovlp=50 \
    --num-hidden-layers=3 \
    --num-encoder-layers=3 \
    --embedding-dim=256 \
    --learning-rate=5e-5 \
    --fp16='False' \
    --deepspeed='none' \
    --use-encoder='True' \
    --do-normalization='True' \
    --run-name='zuco2_custom_training'
```

### Monitor Training Progress

Training logs are saved to `results/models/upstream/<run_name>/`:

```bash
# View training logs
cd NeuroGPT/results/models/upstream/zuco2_custom_training/
cat train_config.json  # Training configuration
ls -lh checkpoint-*/   # Saved checkpoints

# Monitor in real-time (if using a log file)
tail -f trainer_state.json
```

## Step 6: Evaluation and Analysis

### Load Trained Model

```python
import torch
from src.model import Model
from src.train_gpt import make_model, get_config

# Load configuration
config = {...}  # Your training config
model = make_model(config)

# Load checkpoint
checkpoint_path = 'results/models/upstream/zuco2_custom_training/checkpoint-3000/pytorch_model.bin'
model.from_pretrained(checkpoint_path)
model.eval()

print("Model loaded successfully!")
```

### Test on New Data

```python
from src.batcher.zuco_dataset import ZuCo2Dataset

# Create test dataset
test_dataset = ZuCo2Dataset(
    filenames=['path/to/test_file.pt'],
    sample_keys=['inputs', 'attention_mask'],
    chunk_len=500,
    num_chunks=8,
    ovlp=50
)

# Get a sample
sample = test_dataset[0]

# Run inference
with torch.no_grad():
    outputs = model(sample)
    print("Outputs shape:", outputs['outputs'].shape)
```

## Troubleshooting

### Issue: "Cannot load .mat file"

**Solution**: Verify the file is not corrupted:
```bash
python3 -c "import scipy.io; scipy.io.loadmat('file.mat')"
```

### Issue: "Channel mapping error"

**Solution**: ZuCo-2 has different versions. Check your channel layout:
```python
import scipy.io as sio
mat = sio.loadmat('file.mat')
print(mat['channelLabels'])
```

### Issue: "Out of memory during preprocessing"

**Solution**: Process files one at a time:
```bash
# Process each file separately
for file in ~/datasets/zuco2/raw/*.mat; do
    python3 zuco_dataset.py --input "$file" --output ~/datasets/zuco2_preprocessed/
done
```

### Issue: "Very slow preprocessing"

**Expected**: Preprocessing can take 30-60 seconds per file.
- Total time for all subjects: 10-20 minutes
- Be patient or run overnight for large datasets

## Best Practices

1. **Start Small**: Process and train on Task 1 (Normal Reading) first
2. **Verify Each Step**: Check outputs after downloading, preprocessing, and training
3. **Save Intermediate Results**: Keep both raw and preprocessed data
4. **Document Changes**: Note any custom preprocessing steps
5. **Version Control**: Keep track of which ZuCo-2 version you're using

## Performance Expectations (HP Pavilion, Core i5, 16GB RAM)

| Configuration | Preprocessing Time | Training Time (3000 steps) |
|--------------|-------------------|---------------------------|
| 18 subjects, Task 1 | ~15-20 minutes | ~2-3 hours |
| All tasks (54 files) | ~45-60 minutes | ~6-8 hours |

## Additional Resources

- **ZuCo-2 Paper**: https://www.nature.com/articles/sdata2018291
- **Dataset Page**: https://osf.io/2urht/
- **NeuroGPT Paper**: https://arxiv.org/abs/2311.03764
- **EEG Channel System**: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)

## Next Steps

After successful training on ZuCo-2:
1. Evaluate model performance
2. Try different hyperparameters
3. Experiment with different tasks (Task 2, Task 3)
4. Compare with pretrained models
5. Analyze learned representations

## Support

For ZuCo-2 specific issues:
- Check the dataset documentation on OSF
- Verify file integrity
- Ensure you have the correct version

For NeuroGPT issues:
- Refer to CPU_TRAINING_GUIDE.md
- Check the main README.md
- Open an issue on GitHub
