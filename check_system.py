#!/usr/bin/env python3
"""
System Requirements Checker for NeuroGPT CPU Training

This script checks if your system meets the requirements for CPU training.
Useful for HP Pavilion and similar consumer laptops.
"""

import sys
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (Need Python 3.8+)")
        return False


def check_memory():
    """Check system memory."""
    print("\nChecking system memory...")
    
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                mem_total = int([line for line in meminfo.split('\n') 
                               if 'MemTotal' in line][0].split()[1])
                mem_gb = mem_total / (1024 ** 2)
        
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['sysctl', 'hw.memsize'], 
                                  capture_output=True, text=True)
            mem_bytes = int(result.stdout.split()[1])
            mem_gb = mem_bytes / (1024 ** 3)
        
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                ]
            memoryStatus = MEMORYSTATUS()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
            mem_gb = memoryStatus.dwTotalPhys / (1024 ** 3)
        
        else:
            print("  ⚠️  Unknown OS - cannot check memory")
            return True
        
        print(f"  Total RAM: {mem_gb:.1f} GB")
        
        if mem_gb >= 16:
            print(f"  ✓ {mem_gb:.1f} GB RAM (Excellent for CPU training)")
            print(f"    Recommended: 'standard' or 'high_end' config")
            return True
        elif mem_gb >= 8:
            print(f"  ✓ {mem_gb:.1f} GB RAM (Good for CPU training)")
            print(f"    Recommended: 'minimal' or 'standard' config")
            return True
        else:
            print(f"  ⚠️  {mem_gb:.1f} GB RAM (May be insufficient)")
            print(f"    Minimum 8GB recommended. Training may be very slow or fail.")
            print(f"    Use 'minimal' config and close other applications.")
            return False
    
    except Exception as e:
        print(f"  ⚠️  Could not determine memory: {e}")
        return True


def check_cpu():
    """Check CPU information."""
    print("\nChecking CPU...")
    
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        print(f"  CPU cores: {cpu_count}")
        
        if cpu_count >= 4:
            print(f"  ✓ {cpu_count} cores (Good for CPU training)")
            return True
        else:
            print(f"  ⚠️  {cpu_count} cores (Training will be slow)")
            print(f"    Recommended: 4+ cores")
            return False
    
    except Exception as e:
        print(f"  ⚠️  Could not determine CPU: {e}")
        return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\nChecking PyTorch...")
    
    try:
        import torch
        version = torch.__version__
        print(f"  ✓ PyTorch {version} installed")
        
        if torch.cuda.is_available():
            print(f"  ⚠️  CUDA is available - this guide is for CPU training")
            print(f"    You can use GPU training instead for better performance")
        else:
            print(f"  ✓ Running on CPU (CUDA not available)")
        
        return True
    
    except ImportError:
        print(f"  ❌ PyTorch not installed")
        print(f"    Install with: pip install torch torchvision torchaudio")
        return False


def check_dependencies():
    """Check other dependencies."""
    print("\nChecking other dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('h5py', 'h5py'),
        ('transformers', 'transformers'),
        ('tqdm', 'tqdm'),
    ]
    
    all_ok = True
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} not installed")
            all_ok = False
    
    if not all_ok:
        print("\n  Install missing dependencies with:")
        print("    pip install -r requirements.txt")
    
    return all_ok


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    
    try:
        import shutil
        
        current_dir = Path.cwd()
        stat = shutil.disk_usage(current_dir)
        
        free_gb = stat.free / (1024 ** 3)
        print(f"  Free space: {free_gb:.1f} GB")
        
        if free_gb >= 20:
            print(f"  ✓ {free_gb:.1f} GB free (Good)")
            return True
        elif free_gb >= 10:
            print(f"  ⚠️  {free_gb:.1f} GB free (Sufficient for small datasets)")
            return True
        else:
            print(f"  ⚠️  {free_gb:.1f} GB free (May be insufficient)")
            print(f"    Recommended: 10GB+ for datasets and checkpoints")
            return False
    
    except Exception as e:
        print(f"  ⚠️  Could not determine disk space: {e}")
        return True


def main():
    """Run all checks."""
    print("=" * 70)
    print("NeuroGPT CPU Training - System Requirements Check")
    print("=" * 70)
    print()
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.machine()}")
    print()
    
    checks = [
        check_python_version(),
        check_memory(),
        check_cpu(),
        check_pytorch(),
        check_dependencies(),
        check_disk_space(),
    ]
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nPassed {passed}/{total} checks")
    
    if passed == total:
        print("\n✓ System is ready for CPU training!")
        print("\nNext steps:")
        print("  1. Run: python3 example_cpu_training.py")
        print("  2. Read: CPU_TRAINING_GUIDE.md")
        print("  3. Start with a smoke test to verify everything works")
        return 0
    
    elif passed >= total - 1:
        print("\n⚠️  System mostly ready, but with some warnings")
        print("   You can proceed with CPU training, but may need adjustments")
        print("\nNext steps:")
        print("  1. Review warnings above")
        print("  2. Install missing dependencies if any")
        print("  3. Read: CPU_TRAINING_GUIDE.md")
        return 0
    
    else:
        print("\n❌ System has some issues that need to be resolved")
        print("   Please install missing dependencies and check requirements")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure you have at least 8GB RAM")
        print("  3. Re-run this check script")
        return 1


if __name__ == '__main__':
    sys.exit(main())
