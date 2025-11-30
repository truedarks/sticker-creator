"""
Unified background removal module supporting multiple methods:
- rembg CPU
- rembg GPU  
- SAM CPU (Segment Anything Model)
- SAM GPU (Segment Anything Model with GPU)
"""

import os
import sys
from pathlib import Path

# Setup local environment first - MUST be imported before any other imports
from local_env import LOCAL_SITE_PACKAGES
PROJECT_ROOT = Path(__file__).parent.absolute()

# CUDA environment will be setup after function definitions

import traceback
from PIL import Image
import numpy as np
import subprocess

# Enable verbose logging
DEBUG = os.environ.get('STICKER_DEBUG', '0') == '1'

def log_debug(message):
    """Log debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}", file=sys.stderr)

def log_error(message, exc_info=None):
    """Log error message with optional exception info."""
    print(f"[ERROR] {message}", file=sys.stderr)
    if exc_info:
        traceback.print_exception(*exc_info, file=sys.stderr)

# Try to import rembg
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    log_debug("rembg imported successfully")
except ImportError as e:
    REMBG_AVAILABLE = False
    log_debug(f"rembg import failed: {e}")

# Try to import SAM
SAM_AVAILABLE = False
SAM_IMPORT_ERROR = None
try:
    import torch
    import cv2
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
    log_debug("SAM libraries imported successfully")
except ImportError as e:
    SAM_AVAILABLE = False
    SAM_IMPORT_ERROR = str(e)
    log_debug(f"SAM import failed: {e}")

# Check for GPU availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        log_debug(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        log_debug("CUDA not available")
except Exception as e:
    CUDA_AVAILABLE = False
    log_debug(f"CUDA check failed: {e}")


def check_rembg_gpu():
    """Check if rembg GPU (onnxruntime-gpu) is available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        log_debug(f"ONNX Runtime providers: {providers}")
        
        # Check for conflicting packages
        try:
            import pkg_resources
            installed = {pkg.key for pkg in pkg_resources.working_set}
            if 'onnxruntime' in installed and 'onnxruntime-gpu' in installed:
                log_debug("WARNING: Both onnxruntime and onnxruntime-gpu are installed. This often causes GPU to fail.")
                return False # Force fail to trigger re-installation cleanup
        except:
            pass
            
        has_gpu = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        log_debug(f"GPU provider available: {has_gpu}")
        return has_gpu
    except Exception as e:
        log_debug(f"Failed to check rembg GPU: {e}")
        return False


def check_sam_working():
    """Check if SAM actually works (not just imports)."""
    if not SAM_AVAILABLE:
        return False, SAM_IMPORT_ERROR or "SAM libraries not installed"
    
    try:
        from segment_anything import sam_model_registry
        if sam_model_registry is None:
            return False, "SAM model registry is None"
        
        required_models = ['vit_h', 'vit_l', 'vit_b']
        available_models = []
        
        for model_type in required_models:
            try:
                if hasattr(sam_model_registry, 'get'):
                    model_class = sam_model_registry.get(model_type)
                elif isinstance(sam_model_registry, dict):
                    model_class = sam_model_registry.get(model_type)
                elif hasattr(sam_model_registry, model_type):
                    model_class = getattr(sam_model_registry, model_type)
                else:
                    continue
                
                if model_class is not None and callable(model_class):
                    available_models.append(model_type)
            except Exception as e:
                log_debug(f"Error checking SAM model {model_type}: {e}")
                continue
        
        if not available_models:
            return False, "SAM model registry does not contain callable models"
        
        return True, None
    except Exception as e:
        return False, f"SAM check failed: {e}"


def check_cuda_installed():
    """Check if CUDA Toolkit is installed on the system using multiple methods."""
    import subprocess
    import platform
    
    cuda_info = {
        'installed': False,
        'version': None,
        'path': None,
        'drivers_ok': False,
        'cudnn_ok': False,
        'detection_method': None,
        'message': ''
    }
    
    if platform.system() == "Windows":
        # Method 1: Check environment variables
        cuda_env_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_PATH_V11_8', 'CUDA_PATH_V12_1', 'CUDA_PATH_V12_2']
        for env_var in cuda_env_vars:
            cuda_path = os.environ.get(env_var)
            if cuda_path and os.path.exists(cuda_path):
                # Extract version from path or env var name
                version = None
                if 'V11_8' in env_var:
                    version = '11.8'
                elif 'V12_1' in env_var:
                    version = '12.1'
                elif 'V12_2' in env_var:
                    version = '12.2'
                else:
                    # Try to extract version from path
                    path_parts = cuda_path.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part.replace('.', '').isdigit() and len(part.split('.')) >= 2:
                            version = part
                            break
                
                if os.path.exists(os.path.join(cuda_path, 'bin', 'nvcc.exe')):
                    cuda_info['installed'] = True
                    cuda_info['path'] = cuda_path
                    cuda_info['version'] = version or 'unknown'
                    cuda_info['detection_method'] = f'Environment variable {env_var}'
                    cuda_info['message'] = f"CUDA {version or 'unknown'} found via {env_var}: {cuda_path}"
                    log_debug(f"CUDA found via {env_var}: {cuda_path}")
                    break
        
        # Method 2: Check standard installation paths
        if not cuda_info['installed']:
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
                os.path.expanduser(r"~\AppData\Local\NVIDIA\CUDA"),
            ]
            for base_path in cuda_paths:
                if os.path.exists(base_path):
                    try:
                        versions = []
                        for item in os.listdir(base_path):
                            item_path = os.path.join(base_path, item)
                            if os.path.isdir(item_path):
                                # Check if it looks like a version directory
                                if item.replace('.', '').isdigit() or 'v' in item.lower():
                                    # Verify it has CUDA files
                                    if os.path.exists(os.path.join(item_path, 'bin', 'nvcc.exe')):
                                        versions.append(item)
                        
                        if versions:
                            # Get latest version
                            def version_key(v):
                                try:
                                    # Remove 'v' prefix if present
                                    v_clean = v.lstrip('vV')
                                    return [int(i) for i in v_clean.split('.')]
                                except:
                                    return [0, 0]
                            
                            versions.sort(key=version_key, reverse=True)
                            latest_version = versions[0]
                            cuda_path = os.path.join(base_path, latest_version)
                            
                            cuda_info['installed'] = True
                            cuda_info['version'] = latest_version.lstrip('vV')
                            cuda_info['path'] = cuda_path
                            cuda_info['detection_method'] = 'Standard installation path'
                            cuda_info['message'] = f"CUDA {cuda_info['version']} found at {base_path}"
                            log_debug(f"CUDA found at standard path: {cuda_path}")
                            break
                    except Exception as e:
                        log_debug(f"Error checking CUDA path {base_path}: {e}")
                        continue
        
        # Method 3: Check Windows Registry
        if not cuda_info['installed']:
            try:
                import winreg
                registry_paths = [
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\NVIDIA Corporation\CUDA"),
                ]
                
                for hkey, reg_path in registry_paths:
                    try:
                        key = winreg.OpenKey(hkey, reg_path)
                        try:
                            # Try to get version subkeys
                            i = 0
                            while True:
                                try:
                                    subkey_name = winreg.EnumKey(key, i)
                                    subkey = winreg.OpenKey(key, subkey_name)
                                    try:
                                        # Try to get CUDA_PATH value
                                        cuda_path_value, _ = winreg.QueryValueEx(subkey, "CUDA_PATH")
                                        if cuda_path_value and os.path.exists(cuda_path_value):
                                            if os.path.exists(os.path.join(cuda_path_value, 'bin', 'nvcc.exe')):
                                                cuda_info['installed'] = True
                                                cuda_info['path'] = cuda_path_value
                                                cuda_info['version'] = subkey_name
                                                cuda_info['detection_method'] = 'Windows Registry'
                                                cuda_info['message'] = f"CUDA {subkey_name} found via registry: {cuda_path_value}"
                                                log_debug(f"CUDA found via registry: {cuda_path_value}")
                                                break
                                    except:
                                        pass
                                    finally:
                                        winreg.CloseKey(subkey)
                                    i += 1
                                except WindowsError:
                                    break
                        finally:
                            winreg.CloseKey(key)
                        
                        if cuda_info['installed']:
                            break
                    except Exception as e:
                        log_debug(f"Registry check failed for {reg_path}: {e}")
                        continue
            except ImportError:
                log_debug("winreg module not available")
            except Exception as e:
                log_debug(f"Registry check failed: {e}")
        
        # Method 4: Check PATH for nvcc.exe
        if not cuda_info['installed']:
            path_dirs = os.environ.get('PATH', '').split(os.pathsep)
            for path_dir in path_dirs:
                nvcc_path = os.path.join(path_dir, 'nvcc.exe')
                if os.path.exists(nvcc_path):
                    # Found nvcc, try to determine CUDA path
                    cuda_bin_dir = os.path.dirname(nvcc_path)
                    cuda_base_dir = os.path.dirname(cuda_bin_dir)
                    
                    # Try to get version from path
                    version = None
                    path_parts = cuda_base_dir.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part.replace('.', '').isdigit() and len(part.split('.')) >= 2:
                            version = part
                            break
                    
                    cuda_info['installed'] = True
                    cuda_info['path'] = cuda_base_dir
                    cuda_info['version'] = version or 'unknown'
                    cuda_info['detection_method'] = 'PATH environment variable'
                    cuda_info['message'] = f"CUDA {version or 'unknown'} found via PATH: {cuda_base_dir}"
                    log_debug(f"CUDA found via PATH: {cuda_base_dir}")
                    break
        
        # Method 5: Try to run nvcc to get version
        if not cuda_info['installed']:
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, timeout=5, text=True)
                if result.returncode == 0:
                    # Parse version from nvcc output
                    output = result.stdout
                    for line in output.split('\n'):
                        if 'release' in line.lower():
                            import re
                            match = re.search(r'release\s+(\d+\.\d+)', line, re.IGNORECASE)
                            if match:
                                version = match.group(1)
                                # Try to find CUDA path
                                result_path = subprocess.run(['where', 'nvcc'], capture_output=True, timeout=5, text=True)
                                if result_path.returncode == 0:
                                    nvcc_path = result_path.stdout.strip().split('\n')[0]
                                    cuda_bin_dir = os.path.dirname(nvcc_path)
                                    cuda_base_dir = os.path.dirname(cuda_bin_dir)
                                    
                                    cuda_info['installed'] = True
                                    cuda_info['path'] = cuda_base_dir
                                    cuda_info['version'] = version
                                    cuda_info['detection_method'] = 'nvcc command'
                                    cuda_info['message'] = f"CUDA {version} found via nvcc: {cuda_base_dir}"
                                    log_debug(f"CUDA found via nvcc: {cuda_base_dir}")
                                    break
            except Exception as e:
                log_debug(f"nvcc version check failed: {e}")
        
        # Check NVIDIA drivers
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                cuda_info['drivers_ok'] = True
                if not cuda_info['installed']:
                    cuda_info['message'] = "NVIDIA drivers detected but CUDA Toolkit not found"
                else:
                    cuda_info['message'] += " (NVIDIA drivers OK)"
        except Exception as e:
            log_debug(f"nvidia-smi check failed: {e}")
            if not cuda_info['installed']:
                cuda_info['message'] = "CUDA Toolkit not found and NVIDIA drivers not detected"
    
    # Check for CuDNN
    if cuda_info['installed'] and cuda_info['path']:
        try:
            cudnn_path = os.path.join(cuda_info['path'], 'bin', 'cudnn*.dll')
            import glob
            if glob.glob(cudnn_path):
                cuda_info['cudnn_ok'] = True
                log_debug("CuDNN DLLs found in CUDA bin directory")
            else:
                # Check specific paths
                cudnn_path = os.path.join(cuda_info['path'], 'bin', 'cudnn64_8.dll') 
                if os.path.exists(cudnn_path):
                    cuda_info['cudnn_ok'] = True
                    log_debug("CuDNN 8 found")
                else:
                    log_debug("CuDNN DLLs not found in standard path")
                    if 'cudnn' not in cuda_info['message'].lower():
                        cuda_info['message'] += " (Warning: CuDNN not found in CUDA bin)"
        except Exception as e:
            log_debug(f"CuDNN check failed: {e}")
            
    if not cuda_info['installed']:
        cuda_info['message'] = "CUDA Toolkit not found. Checked: environment variables, standard paths, registry, PATH, and nvcc command."
    
    return cuda_info['installed'], cuda_info


def setup_cuda_environment():
    """Setup CUDA environment variables and PATH for local CUDA libraries."""
    # Check for CUDA in local site-packages (from onnxruntime-gpu)
    nvidia_dirs = [
        LOCAL_SITE_PACKAGES / "nvidia" / "cudnn" / "bin",
        LOCAL_SITE_PACKAGES / "nvidia" / "cublas" / "bin",
        LOCAL_SITE_PACKAGES / "nvidia" / "cufft" / "bin",
        LOCAL_SITE_PACKAGES / "nvidia" / "curand" / "bin",
        LOCAL_SITE_PACKAGES / "nvidia" / "cuda_runtime" / "bin",
        LOCAL_SITE_PACKAGES / "nvidia" / "cuda_nvrtc" / "bin",
    ]
    
    cuda_paths_added = []
    for nvidia_dir in nvidia_dirs:
        if nvidia_dir.exists():
            dll_path = str(nvidia_dir.resolve())
            if dll_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
                cuda_paths_added.append(dll_path)
                log_debug(f"Added CUDA path to PATH: {dll_path}")
    
    # Also check system CUDA paths
    cuda_installed, cuda_info = check_cuda_installed()
    if cuda_installed and isinstance(cuda_info, dict) and cuda_info.get('path'):
        cuda_bin = os.path.join(cuda_info['path'], 'bin')
        if os.path.exists(cuda_bin):
            cuda_bin_path = os.path.abspath(cuda_bin)
            if cuda_bin_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')
                cuda_paths_added.append(cuda_bin_path)
                log_debug(f"Added system CUDA path to PATH: {cuda_bin_path}")
    
    return len(cuda_paths_added) > 0


def detect_nvidia_gpu():
    """Detect NVIDIA GPU and check CUDA compatibility."""
    import subprocess
    import platform
    
    gpu_info = {
        'detected': False,
        'name': None,
        'cuda_compute_capability': None,
        'driver_version': None,
        'cuda_supported': False,
        'message': ''
    }
    
    if platform.system() == "Windows":
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'], 
                                  capture_output=True, timeout=5, text=True)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Get first GPU
                    parts = lines[0].split(',')
                    if len(parts) >= 3:
                        gpu_info['detected'] = True
                        gpu_info['name'] = parts[0].strip()
                        gpu_info['driver_version'] = parts[1].strip()
                        compute_cap = parts[2].strip()
                        gpu_info['cuda_compute_capability'] = compute_cap
                        
                        # Check if compute capability supports CUDA (>= 3.0)
                        try:
                            major_version = int(compute_cap.split('.')[0])
                            gpu_info['cuda_supported'] = major_version >= 3
                        except:
                            gpu_info['cuda_supported'] = True  # Assume supported if can't parse
                        
                        gpu_info['message'] = f"NVIDIA GPU detected: {gpu_info['name']} (Driver: {gpu_info['driver_version']}, Compute: {compute_cap})"
        except Exception as e:
            log_debug(f"GPU detection failed: {e}")
            gpu_info['message'] = "Could not detect NVIDIA GPU"
    
    return gpu_info


def get_cuda_version_for_pytorch(cuda_version):
    """Get PyTorch CUDA index URL based on CUDA version."""
    # Map CUDA versions to PyTorch CUDA wheels
    # PyTorch supports CUDA 11.8 and 12.1
    if cuda_version:
        try:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major >= 12:
                return "cu121"  # CUDA 12.1
            elif major == 11 and minor >= 8:
                return "cu118"  # CUDA 11.8
            elif major == 11:
                return "cu118"  # Fallback to 11.8
        except:
            pass
    return "cu118"  # Default to CUDA 11.8


# Setup CUDA environment on import
setup_cuda_environment()


def remove_background_rembg_cpu(image_path, output_path=None):
    """Remove background using rembg with CPU."""
    log_debug(f"Starting rembg CPU processing: {image_path}")
    
    if not REMBG_AVAILABLE:
        error_msg = "rembg is not installed. Install with: pip install rembg onnxruntime"
        log_error(error_msg)
        raise ImportError(error_msg)
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_nobg.png"
    
    try:
        log_debug(f"Reading image file: {image_path}")
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
            log_debug(f"Image size: {len(input_data)} bytes")
        
        log_debug("Calling rembg.remove()...")
        output_data = remove(input_data)
        log_debug(f"rembg returned {len(output_data)} bytes")
        
        log_debug(f"Writing output to: {output_path}")
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
        
        log_debug("Loading result image...")
        result = Image.open(output_path).convert("RGBA")
        log_debug(f"Successfully processed image: {result.size}")
        return result
    except Exception as e:
        error_msg = f"Error removing background with rembg CPU: {e}"
        log_error(error_msg, sys.exc_info())
        raise Exception(error_msg)


def remove_background_rembg_gpu(image_path, output_path=None):
    """Remove background using rembg with GPU acceleration."""
    log_debug(f"Starting rembg GPU processing: {image_path}")
    
    if not REMBG_AVAILABLE:
        error_msg = "rembg is not installed.\n\nTo fix:\n  pip install rembg onnxruntime-gpu\n\nNote: You also need CUDA installed on your system."
        log_error(error_msg)
        raise ImportError(error_msg)
    
    log_debug("Checking GPU availability...")
    gpu_available = check_rembg_gpu()
    if not gpu_available:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            log_debug(f"Available providers: {providers}")
            
            if 'CUDAExecutionProvider' not in providers:
                error_msg = (
                    "GPU not available for rembg.\n\n"
                    "To enable GPU support:\n"
                    "  1. Install: pip install onnxruntime-gpu\n"
                    "  2. Ensure CUDA is installed on your system\n"
                    "  3. Verify GPU drivers are up to date\n\n"
                    "Current providers: " + ", ".join(providers)
                )
            else:
                error_msg = (
                    "GPU provider found but not working.\n\n"
                    "Possible issues:\n"
                    "  - CUDA drivers not installed or outdated\n"
                    "  - GPU not detected by system\n"
                    "  - CUDA version mismatch\n\n"
                    "Try: pip install --upgrade onnxruntime-gpu"
                )
        except Exception as e:
            error_msg = (
                "GPU not available for rembg.\n\n"
                "To enable GPU support:\n"
                "  1. Install: pip install onnxruntime-gpu\n"
                "  2. Ensure CUDA is installed\n\n"
                f"Error checking providers: {e}"
            )
        
        log_error(error_msg)
        raise RuntimeError(error_msg)
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_nobg.png"
    
    try:
        log_debug("Creating rembg session with GPU providers...")
        # STRICTLY use only CUDAExecutionProvider to ensure GPU is used
        # If this fails, we want it to crash rather than silently fall back to CPU
        session = new_session('u2net', providers=['CUDAExecutionProvider'])
        log_debug("Session created successfully")
        
        log_debug(f"Reading image file: {image_path}")
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
            log_debug(f"Image size: {len(input_data)} bytes")
        
        log_debug("Calling rembg.remove() with GPU session...")
        output_data = remove(input_data, session=session)
        log_debug(f"rembg returned {len(output_data)} bytes")
        
        log_debug(f"Writing output to: {output_path}")
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
        
        log_debug("Loading result image...")
        result = Image.open(output_path).convert("RGBA")
        log_debug(f"Successfully processed image with GPU: {result.size}")
        return result
    except Exception as e:
        error_msg = f"Error removing background with rembg GPU: {e}"
        log_error(error_msg, sys.exc_info())
        raise Exception(error_msg)


def remove_background_sam(image_path, output_path=None, model_type='vit_h', checkpoint_path=None, device='cpu'):
    """Remove background using SAM (Segment Anything Model)."""
    log_debug(f"Starting SAM processing: {image_path} (device: {device})")
    
    # Check if SAM is available and working
    sam_working, sam_error = check_sam_working()
    if not sam_working:
        missing = []
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        try:
            import cv2
        except ImportError:
            missing.append("opencv-python")
        
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            missing.append("segment-anything")
        
        if missing:
            error_msg = (
                f"SAM is not installed. Missing: {', '.join(missing)}\n\n"
                "To install SAM:\n"
                "  pip install segment-anything torch torchvision opencv-python\n\n"
                "Note: This will download large model files on first use."
            )
        else:
            error_msg = (
                f"SAM libraries are installed but not working correctly.\n\n"
                f"Error: {sam_error}\n\n"
                "Try:\n"
                "  pip install --upgrade segment-anything torch torchvision opencv-python"
            )
        
        log_error(error_msg)
        raise ImportError(error_msg)
    
    # Override device if CUDA not available
    if device == 'cuda' and not CUDA_AVAILABLE:
        log_debug("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_nobg.png"
    
    try:
        log_debug(f"Loading image with OpenCV: {image_path}")
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        log_debug(f"Image loaded: {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        log_debug(f"Using device: {device}")
        
        # Default checkpoint paths
        if checkpoint_path is None:
            project_checkpoints = PROJECT_ROOT / ".sam_checkpoints"
            project_checkpoints.mkdir(exist_ok=True)
            
            home_dir = os.path.expanduser("~")
            sam_dir = os.path.join(home_dir, ".sam_checkpoints")
            
            checkpoint_urls = {
                'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
            }
            
            checkpoint_filename = f"sam_{model_type}.pth"
            checkpoint_path = project_checkpoints / checkpoint_filename
            if not checkpoint_path.exists():
                checkpoint_path = Path(sam_dir) / checkpoint_filename
                os.makedirs(sam_dir, exist_ok=True)
            
            log_debug(f"SAM checkpoints directory: {checkpoint_path.parent}")
            log_debug(f"Checkpoint path: {checkpoint_path}")
            
            # Download checkpoint if not exists or corrupted
            def download_checkpoint():
                if checkpoint_path.exists():
                    # Check if file is corrupted by trying to load it
                    try:
                        import torch
                        torch.load(str(checkpoint_path), map_location='cpu')
                        log_debug("Checkpoint file exists and is valid")
                        return
                    except Exception as e:
                        log_debug(f"Checkpoint appears corrupted. Re-downloading...")
                        checkpoint_path.unlink()
                
                log_debug(f"Checkpoint not found, downloading {model_type}...")
                print(f"Downloading SAM checkpoint {model_type}...")
                print("This may take a while. Please wait...")
                try:
                    import urllib.request
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(checkpoint_urls[model_type], str(checkpoint_path))
                    log_debug("Checkpoint downloaded successfully")
                    print("Download complete!")
                except Exception as e:
                    error_msg = f"Failed to download SAM checkpoint: {e}\n\nCheck your internet connection and try again."
                    log_error(error_msg, sys.exc_info())
                    raise Exception(error_msg)
            
            download_checkpoint()
        
        log_debug(f"Loading SAM model {model_type} from {checkpoint_path}...")
        try:
            from segment_anything import sam_model_registry
            if isinstance(sam_model_registry, dict):
                model_class = sam_model_registry.get(model_type)
            elif hasattr(sam_model_registry, 'get'):
                model_class = sam_model_registry.get(model_type)
            elif hasattr(sam_model_registry, model_type):
                model_class = getattr(sam_model_registry, model_type)
            else:
                raise ValueError(f"Cannot access SAM model registry. Registry type: {type(sam_model_registry)}")
            
            if model_class is None:
                raise ValueError(f"Unknown SAM model type: {model_type}. Use 'vit_h', 'vit_l', or 'vit_b'")
            
            sam = model_class(checkpoint=str(checkpoint_path))
        except KeyError:
            raise ValueError(f"Unknown SAM model type: {model_type}. Use 'vit_h', 'vit_l', or 'vit_b'")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SAM checkpoint file not found: {checkpoint_path}\n\n"
                "The checkpoint file should be downloaded automatically.\n"
                "If download failed, check your internet connection and try again."
            )
        except Exception as e:
            error_details = str(e)
            if "checkpoint" in error_details.lower() or "file" in error_details.lower():
                raise Exception(
                    f"Failed to load SAM checkpoint: {e}\n\n"
                    f"Checkpoint path: {checkpoint_path}\n"
                    "The checkpoint file might be corrupted. Try deleting it and running again to re-download."
                )
            else:
                raise Exception(
                    f"Failed to load SAM model: {e}\n\n"
                    "This might indicate:\n"
                    "• SAM libraries are not properly installed\n"
                    "• Model checkpoint is corrupted\n"
                    "• Incompatible version of segment-anything\n\n"
                    "Try: pip install --upgrade segment-anything torch torchvision"
                )
        
        log_debug("Model loaded, moving to device...")
        try:
            sam.to(device=device)
        except Exception as e:
            raise Exception(
                f"Failed to move SAM model to device ({device}): {e}\n\n"
                "This might be due to:\n"
                "• Insufficient GPU memory (try using CPU instead)\n"
                "• CUDA/GPU driver issues\n"
                "• PyTorch installation problems"
            )
        log_debug("Model ready")
        
        log_debug("Creating mask generator...")
        try:
            mask_generator = SamAutomaticMaskGenerator(sam)
        except Exception as e:
            raise Exception(
                f"Failed to create mask generator: {e}\n\n"
                "This might indicate:\n"
                "• SAM model is not properly initialized\n"
                "• Incompatible version of segment-anything\n\n"
                "Try: pip install --upgrade segment-anything"
            )
        
        log_debug("Generating masks...")
        try:
            masks = mask_generator.generate(image_rgb)
        except RuntimeError as e:
            error_str = str(e)
            if "out of memory" in error_str.lower() or "cuda" in error_str.lower():
                raise Exception(
                    f"GPU memory error: {e}\n\n"
                    "Solutions:\n"
                    "• Try a smaller image\n"
                    "• Use CPU instead of GPU\n"
                    "• Close other applications using GPU\n"
                    "• Use a smaller SAM model (vit_b instead of vit_h)"
                )
            else:
                raise Exception(
                    f"Failed to generate masks: {e}\n\n"
                    "This might be due to:\n"
                    "• Image size too large\n"
                    "• Memory issues\n"
                    "• SAM model problems\n\n"
                    "Try:\n"
                    "• Resize the image to a smaller size\n"
                    "• Use a different background removal method (rembg CPU)"
                )
        except Exception as e:
            raise Exception(
                f"Failed to generate masks: {e}\n\n"
                "This might be due to:\n"
                "• Image size or memory issues\n"
                "• SAM model problems\n"
                "• Image format issues\n\n"
                "Try:\n"
                "• Use a smaller image\n"
                "• Use rembg CPU instead (more reliable)"
            )
        
        log_debug(f"Generated {len(masks)} masks")
        
        # Find the largest mask (usually the main subject)
        if not masks:
            raise ValueError("No masks generated. Try a different image or check image quality.")
        
        largest_mask = max(masks, key=lambda x: x['area'])
        mask = largest_mask['segmentation']
        log_debug(f"Using largest mask with area: {largest_mask['area']}")
        
        log_debug("Creating RGBA image...")
        image_pil = Image.fromarray(image_rgb).convert("RGBA")
        mask_array = (mask * 255).astype(np.uint8)
        
        # Apply mask to alpha channel
        image_array = np.array(image_pil)
        image_array[:, :, 3] = mask_array
        
        result = Image.fromarray(image_array, 'RGBA')
        log_debug(f"Saving result to: {output_path}")
        result.save(output_path, "PNG")
        log_debug(f"Successfully processed image with SAM: {result.size}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error removing background with SAM: {e}"
        log_error(error_msg, sys.exc_info())
        raise Exception(error_msg)


def normalize_method_name(method):
    """Normalize legacy method names."""
    method = method.lower()
    legacy_map = {
        'sam': 'sam_cpu'
    }
    return legacy_map.get(method, method)


def remove_background(image_path, output_path=None, method='rembg_cpu'):
    """
    Unified function to remove background using specified method.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output (if None, creates _nobg.png suffix)
        method (str): Method to use ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
    
    Returns:
        PIL.Image: Image with background removed, or None on error
    """
    method = normalize_method_name(method)
    log_debug(f"remove_background called with method: {method}, image: {image_path}")
    
    try:
        if method == 'rembg_cpu':
            return remove_background_rembg_cpu(image_path, output_path)
        elif method == 'rembg_gpu':
            return remove_background_rembg_gpu(image_path, output_path)
        elif method == 'sam_cpu':
            return remove_background_sam(image_path, output_path, device='cpu')
        elif method == 'sam_gpu':
            return remove_background_sam(image_path, output_path, device='cuda')
        else:
            error_msg = f"Unknown method: {method}. Choose from: rembg_cpu, rembg_gpu, sam_cpu, sam_gpu"
            log_error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        log_error(f"remove_background failed for method {method}: {e}", sys.exc_info())
        raise


def get_available_methods():
    """Get list of available background removal methods."""
    methods = []
    
    if REMBG_AVAILABLE:
        methods.append('rembg_cpu')
        if check_rembg_gpu():
            methods.append('rembg_gpu')
    
    if SAM_AVAILABLE:
        methods.append('sam_cpu')
        if CUDA_AVAILABLE:
            methods.append('sam_gpu')
    
    return methods if methods else ['rembg_cpu']  # Default fallback


def reload_libraries():
    """Reload libraries and update availability flags after installation."""
    global REMBG_AVAILABLE, SAM_AVAILABLE, CUDA_AVAILABLE, SAM_IMPORT_ERROR
    
    import importlib
    
    # Reload local environment to ensure new packages are in path
    try:
        from local_env import setup_local_environment
        setup_local_environment()
    except:
        pass
    
    # Clear cached modules
    modules_to_clear = ['rembg', 'onnxruntime', 'torch', 'cv2', 'segment_anything']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Try to reimport rembg
    try:
        from rembg import remove, new_session
        REMBG_AVAILABLE = True
        log_debug("rembg reloaded successfully")
    except ImportError as e:
        REMBG_AVAILABLE = False
        log_debug(f"rembg reload failed: {e}")
    
    # Try to reimport SAM
    try:
        import torch
        import cv2
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        SAM_AVAILABLE = True
        SAM_IMPORT_ERROR = None
        log_debug("SAM libraries reloaded successfully")
    except ImportError as e:
        SAM_AVAILABLE = False
        SAM_IMPORT_ERROR = str(e)
        log_debug(f"SAM reload failed: {e}")
    
    # Recheck GPU availability
    try:
        import torch
        CUDA_AVAILABLE = torch.cuda.is_available()
        if CUDA_AVAILABLE:
            log_debug(f"CUDA available after reload: {torch.cuda.get_device_name(0)}")
        else:
            log_debug("CUDA not available after reload")
    except Exception as e:
        CUDA_AVAILABLE = False
        log_debug(f"CUDA check failed after reload: {e}")
    
    # Reload onnxruntime to detect newly installed GPU support
    try:
        if 'onnxruntime' in sys.modules:
            importlib.reload(sys.modules['onnxruntime'])
    except:
        pass


def get_method_status(force_reload=False):
    """Get detailed status of each method.
    
    Args:
        force_reload: If True, reload libraries before checking status
    """
    if force_reload:
        reload_libraries()
    
    # Reload onnxruntime to detect newly installed GPU support
    try:
        import importlib
        if 'onnxruntime' in sys.modules:
            importlib.reload(sys.modules['onnxruntime'])
    except:
        pass
    
    status = {
        'rembg_cpu': {
            'available': REMBG_AVAILABLE,
            'message': 'Available' if REMBG_AVAILABLE else 'Install: pip install rembg onnxruntime'
        },
        'rembg_gpu': {
            'available': False,
            'message': 'Not available'
        },
        'sam_cpu': {
            'available': False,
            'message': 'Not available'
        },
        'sam_gpu': {
            'available': False,
            'message': 'GPU support not detected'
        }
    }
    
    # Check rembg GPU
    if REMBG_AVAILABLE:
        gpu_available = check_rembg_gpu()
        status['rembg_gpu']['available'] = gpu_available
        
        if gpu_available:
            status['rembg_gpu']['message'] = 'Available (GPU detected)'
        else:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                log_debug(f"ONNX Runtime providers for status check: {providers}")
                
                has_gpu_package = False
                has_cpu_package = False
                try:
                    import pkg_resources
                    dists = [d for d in pkg_resources.working_set]
                    for dist in dists:
                        if 'onnxruntime-gpu' in dist.project_name.lower():
                            has_gpu_package = True
                        if 'onnxruntime' == dist.project_name.lower():
                            has_cpu_package = True
                except:
                    pass
                
                if 'CUDAExecutionProvider' not in providers:
                    if has_gpu_package:
                        msg_parts = []
                        if has_cpu_package:
                            msg_parts.append("Conflict: Both 'onnxruntime' and 'onnxruntime-gpu' are installed.")
                            msg_parts.append("The CPU version often overrides the GPU version.")
                            msg_parts.append("Fix: Uninstall both, then install only onnxruntime-gpu.")
                        
                        cuda_installed, cuda_info = check_cuda_installed()
                        if not cuda_installed:
                            cuda_msg = cuda_info.get('message', 'CUDA Toolkit not found') if isinstance(cuda_info, dict) else str(cuda_info)
                            msg_parts.append(f"CUDA Error: {cuda_msg}")
                        else:
                            if isinstance(cuda_info, dict) and not cuda_info.get('cudnn_ok', True):
                                msg_parts.append("Warning: CuDNN libraries might be missing from CUDA bin directory.")
                            
                            cuda_msg = cuda_info.get('message', 'CUDA found') if isinstance(cuda_info, dict) else str(cuda_info)
                            msg_parts.append(f"CUDA Status: {cuda_msg}")
                        
                        status['rembg_gpu']['message'] = "\n".join(msg_parts) if msg_parts else 'onnxruntime-gpu installed but GPU not detected.'
                    else:
                        status['rembg_gpu']['message'] = 'Install: pip install onnxruntime-gpu (and ensure CUDA Toolkit is installed)'
                else:
                    status['rembg_gpu']['message'] = 'GPU provider found but not working (check CUDA installation)'
            except Exception as e:
                status['rembg_gpu']['message'] = f'Install: pip install onnxruntime-gpu (Error: {str(e)[:50]})'
    
    # Check SAM
    if SAM_AVAILABLE:
        sam_working, sam_error = check_sam_working()
        status['sam_cpu']['available'] = sam_working
        status['sam_cpu']['message'] = (
            'Available - Uses Segment Anything Model (CPU)'
            if sam_working else
            (f'SAM libraries not installed: {SAM_IMPORT_ERROR}' if SAM_IMPORT_ERROR else f'SAM installed but not working: {sam_error or "Unknown error"}')
        )
        
        if sam_working and CUDA_AVAILABLE:
            status['sam_gpu']['available'] = True
            status['sam_gpu']['message'] = 'Available - SAM with GPU acceleration'
        else:
            if not sam_working:
                status['sam_gpu']['message'] = status['sam_cpu']['message']
            elif not CUDA_AVAILABLE:
                status['sam_gpu']['message'] = 'CUDA not available. Install torch with CUDA and GPU drivers.'
    else:
        status['sam_cpu']['message'] = 'Install: pip install segment-anything torch torchvision opencv-python'
        status['sam_gpu']['message'] = 'Install: pip install segment-anything torch torchvision opencv-python (with CUDA support)'
    
    return status


def get_method_packages(method):
    """
    Get required packages and their approximate sizes for a method.
    
    Args:
        method (str): Method name ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
    
    Returns:
        dict: {
            'packages': list of package names,
            'total_size_mb': approximate total size in MB,
            'packages_info': list of dicts with 'name' and 'size_mb'
        }
    """
    method = normalize_method_name(method)
    
    # Approximate package sizes in MB (based on typical installations)
    package_sizes = {
        'rembg': 50,  # ~50 MB
        'onnxruntime': 100,  # ~100 MB
        'onnxruntime-gpu': 500,  # ~500 MB (includes CUDA libraries)
        'torch-cpu': 800,  # ~800 MB (CPU version)
        'torch-gpu': 2000,  # ~2 GB (CUDA version)
        'torchvision': 50,  # ~50 MB
        'opencv-python': 50,  # ~50 MB
        'segment-anything': 100,  # ~100 MB (plus model files ~2.4 GB)
    }
    
    # SAM model files sizes
    sam_model_sizes = {
        'vit_h': 2400,  # ~2.4 GB
        'vit_l': 1200,  # ~1.2 GB
        'vit_b': 400,   # ~400 MB
    }
    
    if method == 'rembg_cpu':
        packages = ['rembg', 'onnxruntime']
        total = package_sizes['rembg'] + package_sizes['onnxruntime']
        return {
            'packages': packages,
            'total_size_mb': total,
            'packages_info': [
                {'name': 'rembg', 'size_mb': package_sizes['rembg']},
                {'name': 'onnxruntime', 'size_mb': package_sizes['onnxruntime']}
            ]
        }
    elif method == 'rembg_gpu':
        packages = ['rembg', 'onnxruntime-gpu']
        total = package_sizes['rembg'] + package_sizes['onnxruntime-gpu']
        return {
            'packages': packages,
            'total_size_mb': total,
            'packages_info': [
                {'name': 'rembg', 'size_mb': package_sizes['rembg']},
                {'name': 'onnxruntime-gpu', 'size_mb': package_sizes['onnxruntime-gpu']}
            ],
            'note': 'CUDA Toolkit required (additional ~2-3 GB)'
        }
    elif method == 'sam_cpu':
        packages = ['segment-anything', 'torch', 'torchvision', 'opencv-python']
        total = (package_sizes['segment-anything'] + 
                 package_sizes['torch-cpu'] + 
                 package_sizes['torchvision'] + 
                 package_sizes['opencv-python'] +
                 sam_model_sizes['vit_h'])  # Default model
        return {
            'packages': packages,
            'total_size_mb': total,
            'packages_info': [
                {'name': 'segment-anything', 'size_mb': package_sizes['segment-anything']},
                {'name': 'torch', 'size_mb': package_sizes['torch-cpu']},
                {'name': 'torchvision', 'size_mb': package_sizes['torchvision']},
                {'name': 'opencv-python', 'size_mb': package_sizes['opencv-python']},
                {'name': 'SAM model (vit_h)', 'size_mb': sam_model_sizes['vit_h'], 'download': True}
            ],
            'note': 'Model will be downloaded on first use'
        }
    elif method == 'sam_gpu':
        packages = ['segment-anything', 'torch', 'torchvision', 'opencv-python']
        # GPU version of torch is larger
        total = (package_sizes['segment-anything'] + 
                 package_sizes['torch-gpu'] +  # torch with CUDA support
                 package_sizes['torchvision'] + 
                 package_sizes['opencv-python'] +
                 sam_model_sizes['vit_h'])
        return {
            'packages': packages,
            'total_size_mb': total,
            'packages_info': [
                {'name': 'segment-anything', 'size_mb': package_sizes['segment-anything']},
                {'name': 'torch (CUDA)', 'size_mb': package_sizes['torch-gpu']},
                {'name': 'torchvision', 'size_mb': package_sizes['torchvision']},
                {'name': 'opencv-python', 'size_mb': package_sizes['opencv-python']},
                {'name': 'SAM model (vit_h)', 'size_mb': sam_model_sizes['vit_h'], 'download': True}
            ],
            'note': 'CUDA Toolkit required. Model will be downloaded on first use'
        }
    else:
        return {
            'packages': [],
            'total_size_mb': 0,
            'packages_info': []
        }


def format_size(size_mb):
    """Format size in MB to human-readable format."""
    if size_mb < 1024:
        return f"{size_mb:.0f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"