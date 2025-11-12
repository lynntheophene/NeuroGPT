#!/usr/bin/env python3
"""
Quick Start Example for CPU Training with NeuroGPT

This script demonstrates how to use NeuroGPT on CPU with minimal setup.
Perfect for testing on HP Pavilion or similar consumer laptops.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def main():
    """Run a quick CPU training example."""
    
    print("=" * 70)
    print("NeuroGPT CPU Training - Quick Start Example")
    print("=" * 70)
    print()
    
    # Check if running on CPU
    try:
        import torch
        if torch.cuda.is_available():
            print("⚠️  Warning: GPU detected. This example is for CPU training.")
            print("   The script will still use CPU-optimized settings.")
        else:
            print("✓ Running on CPU (GPU not available)")
        print()
    except ImportError:
        print("❌ PyTorch not installed. Please run: pip install -r requirements.txt")
        return 1
    
    # Show available configurations
    try:
        from cpu_config import get_cpu_config
        
        print("Available CPU Configurations:")
        print("-" * 70)
        
        presets = ['minimal', 'standard', 'high_end', 'quick_test', 
                   'zuco2_minimal', 'zuco2_standard']
        
        for preset in presets:
            config = get_cpu_config(preset)
            print(f"  {preset:20s} - {config['description']}")
        
        print()
        print("Recommended for HP Pavilion: 'standard' or 'zuco2_standard'")
        print()
        
    except ImportError as e:
        print(f"❌ Error importing configuration: {e}")
        return 1
    
    # Show example commands
    print("Example Commands:")
    print("-" * 70)
    print()
    
    print("1. Quick Test (5 minutes, verify everything works):")
    print("   cd src")
    print("   python3 train_gpt.py \\")
    print("       --smoke-test='True' \\")
    print("       --fp16='False' \\")
    print("       --deepspeed='none' \\")
    print("       --num-workers=0")
    print()
    
    print("2. CPU Training with Standard Config:")
    print("   cd scripts")
    print("   ./train_cpu.sh")
    print()
    
    print("3. Training on ZuCo-2 Dataset:")
    print("   # First, preprocess ZuCo-2 data:")
    print("   cd src/batcher")
    print("   python3 zuco_dataset.py \\")
    print("       --input /path/to/zuco2/ \\")
    print("       --output /path/to/zuco2_preprocessed/")
    print()
    print("   # Then train:")
    print("   cd ../../scripts")
    print("   ./finetune_cpu_zuco2.sh /path/to/zuco2_preprocessed/")
    print()
    
    print("4. Custom Configuration:")
    print("   cd src")
    print("   python3 -c \"from cpu_config import get_cpu_config; \\")
    print("                print(get_cpu_config('standard'))\"")
    print()
    
    # Check for data directory
    print("=" * 70)
    print("Data Setup Check:")
    print("-" * 70)
    
    data_paths = [
        '../../tuh_tensors/',
        '../tuh_tensors/',
        './tuh_tensors/',
    ]
    
    found_data = False
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ Found data directory: {path}")
            found_data = True
            break
    
    if not found_data:
        print("⚠️  No data directory found.")
        print("   You need to download and prepare a dataset before training.")
        print()
        print("   Supported datasets:")
        print("   - TUH EEG Corpus: https://isip.piconepress.com/projects/tuh_eeg/")
        print("   - ZuCo-2: https://osf.io/2urht/ (see ZUCO2_SETUP.md)")
        print("   - BCI Competition IV 2a: https://www.bbci.de/competition/iv/")
    
    print()
    print("=" * 70)
    print("Next Steps:")
    print("-" * 70)
    print()
    print("1. Read CPU_TRAINING_GUIDE.md for detailed instructions")
    print("2. Read ZUCO2_SETUP.md for ZuCo-2 dataset setup")
    print("3. Choose a configuration that fits your system RAM")
    print("4. Start with a smoke test to verify everything works")
    print("5. Scale up to full training")
    print()
    print("For help: python3 example_cpu_training.py --help")
    print()
    
    return 0


def show_help():
    """Show help information."""
    print("NeuroGPT CPU Training - Quick Start Example")
    print()
    print("This script provides quick start information for CPU training.")
    print()
    print("Usage:")
    print("  python3 example_cpu_training.py         # Show quick start guide")
    print("  python3 example_cpu_training.py --help  # Show this help")
    print()
    print("Documentation:")
    print("  CPU_TRAINING_GUIDE.md  - Complete CPU training guide")
    print("  ZUCO2_SETUP.md         - ZuCo-2 dataset setup guide")
    print("  README.md              - Main project README")
    print()
    print("Support:")
    print("  GitHub Issues: https://github.com/lynntheophene/NeuroGPT/issues")
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    sys.exit(main())
