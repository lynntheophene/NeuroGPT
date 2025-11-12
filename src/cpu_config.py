#!/usr/bin/env python3
"""
CPU-Optimized Configuration Presets for NeuroGPT

This module provides pre-configured settings for running NeuroGPT on CPU
with different hardware constraints.

Usage:
    from cpu_config import get_cpu_config
    
    # For minimal memory usage (8GB RAM)
    config = get_cpu_config('minimal')
    
    # For standard laptops (16GB RAM)
    config = get_cpu_config('standard')
    
    # For high-end systems (32GB+ RAM)
    config = get_cpu_config('high_end')
"""

def get_cpu_config(preset='standard'):
    """
    Get CPU-optimized configuration preset.
    
    Args:
        preset (str): Configuration preset name
            - 'minimal': For systems with 8GB RAM (very conservative)
            - 'standard': For typical laptops with 16GB RAM (recommended)
            - 'high_end': For high-end systems with 32GB+ RAM
            - 'quick_test': For quick testing (smoke test-like)
    
    Returns:
        dict: Configuration dictionary for train_gpt.py
    """
    
    # Base configuration that applies to all CPU setups
    base_config = {
        'fp16': 'False',                    # CPU doesn't support FP16
        'deepspeed': 'none',                # Deepspeed requires GPU
        'num_workers': 0,                   # Avoid multiprocessing overhead
        'do_normalization': 'True',
        'use_encoder': 'True',
        'training_style': 'CSM_causal',
        'architecture': 'GPT',
        'chunk_len': 500,
        'sampling_rate': 250,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'lr_scheduler': 'linear',
        'warmup_ratio': 0.01,
    }
    
    # Preset-specific configurations
    presets = {
        'minimal': {
            # For HP Pavilion or similar with 8GB RAM
            # Estimated memory usage: ~4-6GB
            # Training speed: ~5-10 seconds/step
            'per_device_training_batch_size': 1,
            'per_device_validation_batch_size': 1,
            'num_chunks': 4,
            'chunk_ovlp': 25,
            'num_hidden_layers': 2,
            'num_encoder_layers': 2,
            'embedding_dim': 128,
            'num_attention_heads': 2,
            'training_steps': 2000,
            'eval_every_n_steps': 500,
            'log_every_n_steps': 500,
            'description': 'Minimal config for 8GB RAM systems'
        },
        
        'standard': {
            # For HP Pavilion with 16GB RAM (recommended)
            # Estimated memory usage: ~6-10GB
            # Training speed: ~2-4 seconds/step
            'per_device_training_batch_size': 2,
            'per_device_validation_batch_size': 2,
            'num_chunks': 8,
            'chunk_ovlp': 50,
            'num_hidden_layers': 3,
            'num_encoder_layers': 3,
            'embedding_dim': 256,
            'num_attention_heads': 4,
            'training_steps': 5000,
            'eval_every_n_steps': 500,
            'log_every_n_steps': 500,
            'description': 'Standard config for 16GB RAM systems'
        },
        
        'high_end': {
            # For high-end systems with 32GB+ RAM
            # Estimated memory usage: ~12-20GB
            # Training speed: ~1-2 seconds/step
            'per_device_training_batch_size': 4,
            'per_device_validation_batch_size': 4,
            'num_chunks': 16,
            'chunk_ovlp': 50,
            'num_hidden_layers': 4,
            'num_encoder_layers': 4,
            'embedding_dim': 512,
            'num_attention_heads': 8,
            'training_steps': 10000,
            'eval_every_n_steps': 500,
            'log_every_n_steps': 500,
            'description': 'High-end config for 32GB+ RAM systems'
        },
        
        'quick_test': {
            # For quick testing (completes in ~5 minutes)
            # Minimal training, just to verify everything works
            'per_device_training_batch_size': 2,
            'per_device_validation_batch_size': 2,
            'num_chunks': 4,
            'chunk_ovlp': 25,
            'num_hidden_layers': 2,
            'num_encoder_layers': 2,
            'embedding_dim': 128,
            'num_attention_heads': 2,
            'training_steps': 100,
            'eval_every_n_steps': 50,
            'log_every_n_steps': 50,
            'description': 'Quick test config (~5 minutes)'
        },
        
        'zuco2_minimal': {
            # Optimized for ZuCo-2 dataset on 8GB RAM
            'per_device_training_batch_size': 1,
            'per_device_validation_batch_size': 1,
            'num_chunks': 4,
            'chunk_ovlp': 50,
            'num_hidden_layers': 2,
            'num_encoder_layers': 2,
            'embedding_dim': 128,
            'num_attention_heads': 2,
            'training_steps': 2000,
            'eval_every_n_steps': 200,
            'log_every_n_steps': 200,
            'learning_rate': 5e-5,  # Lower for fine-tuning
            'description': 'ZuCo-2 optimized for 8GB RAM'
        },
        
        'zuco2_standard': {
            # Optimized for ZuCo-2 dataset on 16GB RAM
            'per_device_training_batch_size': 2,
            'per_device_validation_batch_size': 2,
            'num_chunks': 8,
            'chunk_ovlp': 50,
            'num_hidden_layers': 3,
            'num_encoder_layers': 3,
            'embedding_dim': 256,
            'num_attention_heads': 4,
            'training_steps': 3000,
            'eval_every_n_steps': 300,
            'log_every_n_steps': 300,
            'learning_rate': 5e-5,  # Lower for fine-tuning
            'description': 'ZuCo-2 optimized for 16GB RAM'
        }
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")
    
    # Merge base config with preset
    config = {**base_config, **presets[preset]}
    
    return config


def get_hp_pavilion_config():
    """
    Get recommended configuration for HP Pavilion laptops.
    
    This is an alias for the 'standard' preset, optimized for typical
    HP Pavilion specs (Core i5/i7, 16GB RAM).
    
    Returns:
        dict: Configuration dictionary
    """
    return get_cpu_config('standard')


def print_config_info():
    """Print information about all available presets."""
    presets = ['minimal', 'standard', 'high_end', 'quick_test', 
               'zuco2_minimal', 'zuco2_standard']
    
    print("Available CPU Configuration Presets:")
    print("=" * 70)
    
    for preset in presets:
        config = get_cpu_config(preset)
        print(f"\n{preset.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Batch Size: {config['per_device_training_batch_size']}")
        print(f"  Model Layers: {config['num_hidden_layers']}")
        print(f"  Embedding Dim: {config['embedding_dim']}")
        print(f"  Training Steps: {config['training_steps']}")
        
        # Estimate memory
        mem_est = (
            config['per_device_training_batch_size'] * 
            config['num_hidden_layers'] * 
            config['embedding_dim'] * 2
        )
        mem_gb = mem_est / (1024 ** 3) * 50  # Rough estimate
        print(f"  Est. Memory: ~{mem_gb:.1f}GB")


def create_training_command(preset='standard', data_path='../../tuh_tensors/', 
                           output_dir='results/models/upstream', 
                           run_name=None, pretrained_model=None):
    """
    Create a training command with the specified preset.
    
    Args:
        preset (str): Configuration preset name
        data_path (str): Path to training data
        output_dir (str): Output directory for logs and checkpoints
        run_name (str): Name for this training run
        pretrained_model (str): Path to pretrained model (optional)
    
    Returns:
        str: Command string ready to execute
    """
    config = get_cpu_config(preset)
    
    if run_name is None:
        run_name = f'cpu_{preset}'
    
    cmd_parts = ['python3 train_gpt.py']
    
    # Add all config parameters
    for key, value in config.items():
        if key == 'description':
            continue
        # Convert underscores to hyphens for command line
        cmd_key = key.replace('_', '-')
        cmd_parts.append(f'--{cmd_key}={value}')
    
    # Add paths
    cmd_parts.append(f'--train-data-path={data_path}')
    cmd_parts.append(f'--log-dir={output_dir}')
    cmd_parts.append(f'--run-name={run_name}')
    
    if pretrained_model:
        cmd_parts.append(f'--pretrained-model={pretrained_model}')
    
    return ' \\\n    '.join(cmd_parts)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        preset = sys.argv[1]
        
        if preset == '--list':
            print_config_info()
        elif preset == '--command':
            # Generate a command
            preset_name = sys.argv[2] if len(sys.argv) > 2 else 'standard'
            data_path = sys.argv[3] if len(sys.argv) > 3 else '../../tuh_tensors/'
            
            print(f"\n# Training command for {preset_name} preset:")
            print(create_training_command(preset_name, data_path))
        else:
            # Show specific preset config
            config = get_cpu_config(preset)
            print(f"\nConfiguration for '{preset}' preset:")
            print("=" * 60)
            for key, value in config.items():
                print(f"{key:35s}: {value}")
    else:
        print("Usage:")
        print("  python3 cpu_config.py --list                    # List all presets")
        print("  python3 cpu_config.py <preset>                  # Show preset config")
        print("  python3 cpu_config.py --command <preset> <path> # Generate command")
        print("\nExample:")
        print("  python3 cpu_config.py standard")
        print("  python3 cpu_config.py --command zuco2_standard ~/datasets/zuco2/")
