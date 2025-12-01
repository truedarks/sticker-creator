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
import json
from datetime import datetime

# Enable verbose logging to stderr via env var
DEBUG = os.environ.get('STICKER_DEBUG', '0') == '1'

# Application-level logging to file, controlled by app settings
APP_CONFIG_DIR = PROJECT_ROOT / ".config"
APP_CONFIG_DIR.mkdir(exist_ok=True)
APP_CONFIG_PATH = APP_CONFIG_DIR / "app_config.json"
LOG_FILE_PATH = APP_CONFIG_DIR / "sticker_creator.log"

def _load_log_to_file_flag():
    """Load 'save_logs' flag from general app settings file."""
    try:
        if APP_CONFIG_PATH.exists():
            with open(APP_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return bool(data.get("save_logs", False))
    except Exception:
        # Fail silently – logging to file is optional
        pass
    return False

LOG_TO_FILE = _load_log_to_file_flag()

def _write_log_line(prefix, message):
    """Write single log line to file if enabled."""
    if not LOG_TO_FILE:
        return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {prefix} {message}\n")
    except Exception:
        # Do not break processing if logging fails
        pass

def log_debug(message):
    """Log debug message (stderr when DEBUG, and optionally to file)."""
    text = f"[DEBUG] {message}"
    if DEBUG:
        print(text, file=sys.stderr)
    _write_log_line("DEBUG", message)

def log_error(message, exc_info=None):
    """Log error message with optional exception info."""
    text = f"[ERROR] {message}"
    print(text, file=sys.stderr)
    _write_log_line("ERROR", message)
    if exc_info:
        traceback.print_exception(*exc_info, file=sys.stderr)

# Try to import LLM censor
try:
    from llm_censor import create_llm_censor, OllamaLLMCensor
    LLM_CENSOR_AVAILABLE = True
    log_debug("LLM censor module imported successfully")
except ImportError as e:
    LLM_CENSOR_AVAILABLE = False
    log_debug(f"LLM censor import failed: {e}")

# Global cache for SAM model to avoid reloading for each image
_SAM_MODEL_CACHE = {
    'model': None,
    'mask_generator': None,
    'model_type': None,
    'device': None,
    'checkpoint_path': None,
    'generator_params': None  # Generator parameters for caching
}

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
        error_str = str(e).lower()
        error_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
        
        # Check for access violation or critical errors
        is_critical = (
            'access violation' in error_str or
            '0xc0000005' in error_str or
            'status_access_violation' in error_str or
            error_code == 3221226505 or
            'cuda' in error_str and ('error' in error_str or 'failed' in error_str)
        )
        
        if is_critical:
            error_msg = (
                f"Critical error removing background with rembg GPU: {e}\n\n"
                "This is likely a GPU/CUDA driver issue.\n\n"
                "Possible causes:\n"
                "  • Outdated or corrupted NVIDIA drivers\n"
                "  • CUDA Toolkit not properly installed\n"
                "  • Missing or incompatible CuDNN libraries\n"
                "  • Library conflicts (onnxruntime vs onnxruntime-gpu)\n"
                "  • Insufficient GPU memory\n\n"
                "Solutions:\n"
                "  1. Try using rembg_cpu method instead\n"
                "  2. Update NVIDIA drivers to latest version\n"
                "  3. Reinstall onnxruntime-gpu: pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu\n"
                "  4. Verify CUDA installation: nvidia-smi\n"
                "  5. Restart the application\n"
            )
        else:
            error_msg = f"Error removing background with rembg GPU: {e}"
        
        log_error(error_msg, sys.exc_info())
        raise Exception(error_msg)


def get_cached_sam_model(model_type='vit_h', checkpoint_path=None, device='cpu', generator_params=None):
    """
    Get cached SAM model or load a new one if needed.
    This significantly speeds up processing multiple images.
    
    Args:
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: Path to checkpoint (None = auto-download)
        device: Device ('cpu' or 'cuda')
        generator_params: Parameters for SamAutomaticMaskGenerator (None = default values)
    """
    global _SAM_MODEL_CACHE
    
    # Default generator parameters
    if generator_params is None:
        generator_params = {
            'points_per_side': 32,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
            'min_mask_region_area': 100,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'box_nms_thresh': 0.7,
        }
    
    # Check if we have a cached model with the same parameters
    # Note: if checkpoint_path is None, we only compare model_type and device
    # because the actual path is computed inside this function
    cache_match = (
        _SAM_MODEL_CACHE['model'] is not None and
        _SAM_MODEL_CACHE['model_type'] == model_type and
        _SAM_MODEL_CACHE['device'] == device and
        _SAM_MODEL_CACHE['generator_params'] == generator_params
    )
    
    # If explicit checkpoint_path provided, also verify it matches
    if cache_match and checkpoint_path is not None:
        cache_match = _SAM_MODEL_CACHE['checkpoint_path'] == str(checkpoint_path)
    
    if cache_match:
        log_debug(f"Using cached SAM model (type: {model_type}, device: {device})")
        return _SAM_MODEL_CACHE['model'], _SAM_MODEL_CACHE['mask_generator']
    
    log_debug(f"Loading new SAM model (type: {model_type}, device: {device})")
    
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
        if checkpoint_path.exists():
            # Check if file is corrupted by trying to load it
            try:
                import torch
                torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
                log_debug("Checkpoint file exists and is valid")
            except Exception as e:
                log_debug(f"Checkpoint appears corrupted. Re-downloading...")
                checkpoint_path.unlink()
        
        if not checkpoint_path.exists():
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
    
    log_debug(f"Loading SAM model {model_type} from {checkpoint_path}...")
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
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
        # Create generator with specified parameters
        mask_generator = SamAutomaticMaskGenerator(sam, **generator_params)
        log_debug(f"Mask generator created with params: {generator_params}")
    except Exception as e:
        raise Exception(
            f"Failed to create mask generator: {e}\n\n"
            "This might indicate:\n"
            "• SAM model is not properly initialized\n"
            "• Incompatible version of segment-anything\n"
            "• Invalid generator parameters\n\n"
            "Try: pip install --upgrade segment-anything"
        )
    
    # Cache the model for future use
    _SAM_MODEL_CACHE['model'] = sam
    _SAM_MODEL_CACHE['mask_generator'] = mask_generator
    _SAM_MODEL_CACHE['model_type'] = model_type
    _SAM_MODEL_CACHE['device'] = device
    _SAM_MODEL_CACHE['checkpoint_path'] = str(checkpoint_path)
    _SAM_MODEL_CACHE['generator_params'] = generator_params
    
    log_debug(f"SAM model cached (type: {model_type}, device: {device}, params: {generator_params})")
    
    return sam, mask_generator


def clear_sam_cache():
    """Clear the SAM model cache to free memory."""
    global _SAM_MODEL_CACHE
    if _SAM_MODEL_CACHE['model'] is not None:
        log_debug("Clearing SAM model cache...")
        _SAM_MODEL_CACHE['model'] = None
        _SAM_MODEL_CACHE['mask_generator'] = None
        _SAM_MODEL_CACHE['model_type'] = None
        _SAM_MODEL_CACHE['device'] = None
        _SAM_MODEL_CACHE['checkpoint_path'] = None
        _SAM_MODEL_CACHE['generator_params'] = None
        
        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_debug("GPU cache cleared")
        except:
            pass


def get_saved_sam_params_path():
    """Get path to saved SAM parameters file."""
    params_dir = PROJECT_ROOT / ".sam_params"
    params_dir.mkdir(exist_ok=True)
    return params_dir / "successful_params.json"


def load_saved_sam_params():
    """Load saved successful SAM parameters from file."""
    params_path = get_saved_sam_params_path()
    if not params_path.exists():
        return None
    
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
            params = data.get('params', {})
            log_debug(f"Loaded saved SAM parameters: {params}")
            return params
    except Exception as e:
        log_debug(f"Error loading saved SAM parameters: {e}")
        return None


def save_sam_params(params):
    """Save successful SAM parameters to file."""
    params_path = get_saved_sam_params_path()
    try:
        data = {
            'params': params,
            'timestamp': str(Path(__file__).stat().st_mtime)  # Simple timestamp
        }
        with open(params_path, 'w') as f:
            json.dump(data, f, indent=2)
        log_debug(f"Saved successful SAM parameters: {params}")
    except Exception as e:
        log_debug(f"Error saving SAM parameters: {e}")


def remove_background_sam(image_path, output_path=None, model_type='vit_h', checkpoint_path=None, device='cpu', llm_censor=None, max_iterations=3, save_successful_params=False, llm_debug=False):
    """
    Remove background using SAM (Segment Anything Model) with optional LLM censor.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output (if None, creates _nobg.png suffix)
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: Path to SAM checkpoint (None = auto-download)
        device: Device ('cpu' or 'cuda')
        llm_censor: LLM censor instance for quality control (None = disabled)
        max_iterations: Maximum number of iterations for parameter tuning (default: 3)
        save_successful_params: Whether to save successful parameters for next time (default: False)
        llm_debug: Debug mode - keep all intermediate results numbered and return debug info (default: False)
    
    Returns:
        PIL.Image: Image with background removed, or dict with debug info if llm_debug=True
    """
    log_debug(f"Starting SAM processing: {image_path} (device: {device}, llm_censor: {llm_censor is not None})")
    
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
    
    # 1. Parallel analysis of original image by LLM (if enabled)
    image_analysis = None
    if llm_censor and llm_censor.enabled:
        try:
            log_debug("LLM censor: analyzing original image...")
            image_analysis = llm_censor.analyze_image(image_path)
            if image_analysis:
                log_debug(f"LLM analysis: {image_analysis}")
        except Exception as e:
            log_debug(f"Error analyzing image with LLM: {e}. Continuing without LLM analysis.")
    
    # Initial SAM generator parameters
    # Try to load saved successful parameters first
    # Use saved params if: (save_successful_params enabled OR debug mode disabled) AND saved params exist
    saved_params = None
    if (save_successful_params or not llm_debug):
        saved_params = load_saved_sam_params()
        if saved_params:
            log_debug(f"Using saved winning parameters as starting point: {saved_params}")
            print(f"[SAM] Using saved winning parameters from previous debug session", file=sys.stderr)
    
    # Use saved parameters if available, otherwise use defaults
    default_params = {
        'points_per_side': 32,
        'pred_iou_thresh': 0.88,
        'stability_score_thresh': 0.95,
        'min_mask_region_area': 100,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'box_nms_thresh': 0.7,
    }
    
    # Merge saved params with defaults (in case new params were added)
    if saved_params:
        sam_params = {**default_params, **saved_params}
    else:
        sam_params = default_params.copy()
    
    # Initialize variables before try block so they're available in except blocks
    best_result = None
    best_score = 0
    all_results = []  # List of (result, score, path, iteration) tuples
    result = None  # Last result from iteration
    debug_results_list = []  # For debug mode: list of dicts with full info
    
    try:
        log_debug(f"Loading image: {image_path}")
        import cv2
        import numpy as np
        
        # OpenCV imread doesn't work with Unicode paths on Windows
        # Use alternative method: read file as bytes and decode with cv2.imdecode
        try:
            # Try direct path first (works for ASCII paths)
            image = cv2.imread(image_path)
            if image is None:
                # If failed, try reading file as bytes (works with Unicode paths)
                log_debug("Direct cv2.imread failed, trying byte reading method for Unicode path...")
                with open(image_path, 'rb') as f:
                    image_bytes = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            # Fallback: use PIL to load image (handles Unicode paths well)
            log_debug(f"OpenCV loading failed ({e}), trying PIL...")
            from PIL import Image as PILImage
            pil_image = PILImage.open(image_path)
            # Convert PIL image to OpenCV format (BGR)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        log_debug(f"Image loaded: {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        log_debug(f"Using device: {device}")
        
        for iteration in range(max_iterations):
            log_debug(f"Iteration {iteration + 1}/{max_iterations} with parameters: {sam_params}")
            current_iteration_params = sam_params.copy()  # Track parameters for this iteration
            current_iteration_params = sam_params.copy()  # Track parameters for this iteration
            
            # Clear generator cache if parameters changed
            if iteration > 0:
                clear_sam_cache()
            
            # Get cached SAM model (or load if not cached) with current parameters
            sam, mask_generator = get_cached_sam_model(model_type, checkpoint_path, device, sam_params)
            
            log_debug("Generating masks...")
            # Warn user if this might take a while (large images or GPU)
            if device == 'cuda':
                log_debug("Using GPU - this may take several minutes for large images")
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
            
            if not masks:
                raise ValueError("No masks generated. Try a different image or check image quality.")
            
            # Sort masks by predicted_iou (highest first) and then by area (largest first)
            # This prioritizes masks with higher confidence over just size
            sorted_masks = sorted(masks, key=lambda x: (x.get('predicted_iou', 0), x.get('stability_score', 0), x['area']), reverse=True)
            
            # Also try smallest masks (in case object is small and background is large)
            sorted_by_area_asc = sorted(masks, key=lambda x: x['area'])
            smallest_masks = sorted_by_area_asc[:min(3, len(sorted_by_area_asc))]
            
            # Combine top masks with smallest masks (remove duplicates)
            # Use id() to compare mask objects instead of comparing dictionaries with NumPy arrays
            candidate_masks = sorted_masks[:min(6, len(sorted_masks))]
            candidate_mask_ids = {id(m) for m in candidate_masks}
            for small_mask in smallest_masks:
                if id(small_mask) not in candidate_mask_ids:
                    candidate_masks.append(small_mask)
                    candidate_mask_ids.add(id(small_mask))
            
            # Limit to reasonable number
            num_candidates = min(8, len(candidate_masks))
            candidate_masks = candidate_masks[:num_candidates]
            
            selected_mask_idx = 0  # Default to first (largest/highest iou)
            result = None
            temp_output = None
            
            # If LLM censor is enabled, let it choose the best mask from candidates
            if llm_censor and llm_censor.enabled and len(candidate_masks) > 1:
                log_debug(f"LLM censor: evaluating {len(candidate_masks)} mask candidates...")
                
                # Create temporary results for each candidate
                candidate_results = []
                image_pil = Image.fromarray(image_rgb).convert("RGBA")
                
                for idx, candidate_mask in enumerate(candidate_masks):
                    mask = candidate_mask['segmentation']
                    
                    # Add normal mask
                    mask_array = (mask * 255).astype(np.uint8)
                    image_array = np.array(image_pil.copy())
                    image_array[:, :, 3] = mask_array
                    result_img = Image.fromarray(image_array, 'RGBA')
                    temp_candidate_path = f"{output_path}.candidate_{iteration}_{idx}.png"
                    result_img.save(temp_candidate_path, "PNG")
                    
                    candidate_results.append({
                        'mask_index': idx,
                        'result_path': temp_candidate_path,
                        'area': candidate_mask['area'],
                        'predicted_iou': candidate_mask.get('predicted_iou', 0),
                        'stability_score': candidate_mask.get('stability_score', 0),
                        'is_inverted': False
                    })
                    
                    # In debug mode, add all candidates to debug_results_list for voting
                    if llm_debug:
                        result_num = len(debug_results_list) + 1
                        debug_results_list.append({
                            'index': len(debug_results_list),
                            'number': result_num,
                            'path': temp_candidate_path,
                            'score': candidate_mask.get('predicted_iou', 0) * 10,  # Convert iou (0-1) to score (0-10)
                            'iteration': iteration,
                            'params': current_iteration_params.copy(),
                            'approved': False,
                            'wrong_selection': False,
                            'is_inverted': False,
                            'is_candidate': True,
                            'candidate_index': idx,
                            'area': candidate_mask['area'],
                            'predicted_iou': candidate_mask.get('predicted_iou', 0),
                            'stability_score': candidate_mask.get('stability_score', 0)
                        })
                        log_debug(f"Added candidate #{idx} (normal) to debug results list as option #{result_num}")
                    
                    # Also add inverted mask as candidate (in case object is smaller than background)
                    # Ensure mask is boolean before inverting
                    if mask.dtype != bool:
                        mask_bool = mask.astype(bool)
                    else:
                        mask_bool = mask
                    inverted_mask = ~mask_bool
                    mask_array_inv = (inverted_mask.astype(np.uint8) * 255).astype(np.uint8)
                    image_array_inv = np.array(image_pil.copy())
                    image_array_inv[:, :, 3] = mask_array_inv
                    result_img_inv = Image.fromarray(image_array_inv, 'RGBA')
                    temp_candidate_path_inv = f"{output_path}.candidate_{iteration}_{idx}_inv.png"
                    result_img_inv.save(temp_candidate_path_inv, "PNG")
                    
                    candidate_results.append({
                        'mask_index': idx,
                        'result_path': temp_candidate_path_inv,
                        'area': image_rgb.shape[0] * image_rgb.shape[1] - candidate_mask['area'],  # Inverted area
                        'predicted_iou': candidate_mask.get('predicted_iou', 0),
                        'stability_score': candidate_mask.get('stability_score', 0),
                        'is_inverted': True
                    })
                    
                    # In debug mode, add inverted candidate to debug_results_list for voting
                    if llm_debug:
                        result_num = len(debug_results_list) + 1
                        debug_results_list.append({
                            'index': len(debug_results_list),
                            'number': result_num,
                            'path': temp_candidate_path_inv,
                            'score': candidate_mask.get('predicted_iou', 0) * 10,  # Convert iou (0-1) to score (0-10)
                            'iteration': iteration,
                            'params': current_iteration_params.copy(),
                            'approved': False,
                            'wrong_selection': False,
                            'is_inverted': True,
                            'is_candidate': True,
                            'candidate_index': idx,
                            'area': image_rgb.shape[0] * image_rgb.shape[1] - candidate_mask['area'],
                            'predicted_iou': candidate_mask.get('predicted_iou', 0),
                            'stability_score': candidate_mask.get('stability_score', 0)
                        })
                        log_debug(f"Added candidate #{idx} (inverted) to debug results list as option #{result_num}")
                
                # Let LLM choose the best mask
                # Note: Timeout is handled inside llm_censor (requests timeout=300)
                # If LLM hangs, it will raise an exception which we catch
                try:
                    best_idx = llm_censor.select_best_mask(image_path, candidate_results)
                except Exception as e:
                    log_error(f"LLM mask selection failed or timed out: {e}. Using first candidate.")
                    best_idx = 0  # Fallback to first candidate
                if best_idx is not None and 0 <= best_idx < len(candidate_results):
                    selected_result = candidate_results[best_idx]
                    selected_mask_idx = selected_result['mask_index']
                    is_inverted = selected_result.get('is_inverted', False)
                    
                    log_debug(f"LLM selected mask candidate {selected_mask_idx} (inverted: {is_inverted}, area: {selected_result['area']}, iou: {selected_result['predicted_iou']})")
                    
                    # Use the selected result file directly
                    result = Image.open(selected_result['result_path']).convert('RGBA')
                    temp_output = selected_result['result_path']
                    
                    # Get the mask for potential later use
                    if is_inverted:
                        orig_mask = candidate_masks[selected_mask_idx]['segmentation']
                        # Ensure mask is boolean before inverting
                        if orig_mask.dtype != bool:
                            mask_bool = orig_mask.astype(bool)
                        else:
                            mask_bool = orig_mask
                        mask = ~mask_bool
                    else:
                        mask = candidate_masks[selected_mask_idx]['segmentation']
                else:
                    log_debug("LLM mask selection failed, using first candidate")
                    selected_mask = candidate_masks[0]
                    mask = selected_mask['segmentation']
                    mask_array = (mask * 255).astype(np.uint8)
                    image_array = np.array(image_pil)
                    image_array[:, :, 3] = mask_array
                    result = Image.fromarray(image_array, 'RGBA')
                    temp_output = candidate_results[0]['result_path']
                
                # Clean up other candidate files (but keep the selected one)
                # In debug mode, keep all candidate files
                if not llm_debug:
                    for idx, candidate_result in enumerate(candidate_results):
                        if idx != best_idx:
                            try:
                                os.remove(candidate_result['result_path'])
                            except:
                                pass
                
                # Save to standard temp name for evaluation
                # IMPORTANT: Save best result before any cleanup
                final_temp = output_path if iteration == max_iterations - 1 else f"{output_path}.temp_{iteration}.png"
                if temp_output != final_temp:
                    # Save result to final temp location
                    result.save(final_temp, "PNG")
                    # Only remove old temp if it's different from the selected candidate
                    # In debug mode, keep all temp files
                    if not llm_debug and temp_output != selected_result['result_path']:
                        try:
                            os.remove(temp_output)
                        except:
                            pass
                    temp_output = final_temp
                else:
                    # If temp_output is already the final temp, make sure it's saved
                    result.save(final_temp, "PNG")
            else:
                # No LLM censor or only one candidate - use largest/highest iou
                selected_mask = candidate_masks[0]
                mask = selected_mask['segmentation']
                log_debug(f"Using mask with area: {selected_mask['area']}, predicted_iou: {selected_mask.get('predicted_iou', 0)}")
                
                log_debug("Creating RGBA image...")
                image_pil = Image.fromarray(image_rgb).convert("RGBA")
                
                # In debug mode, add all candidates to debug_results_list for voting
                if llm_debug:
                    for idx, candidate_mask in enumerate(candidate_masks):
                        mask_cand = candidate_mask['segmentation']
                        
                        # Normal mask
                        mask_array_cand = (mask_cand * 255).astype(np.uint8)
                        image_array_cand = np.array(image_pil.copy())
                        image_array_cand[:, :, 3] = mask_array_cand
                        result_img_cand = Image.fromarray(image_array_cand, 'RGBA')
                        temp_candidate_path = f"{output_path}.candidate_{iteration}_{idx}.png"
                        result_img_cand.save(temp_candidate_path, "PNG")
                        
                        result_num = len(debug_results_list) + 1
                        debug_results_list.append({
                            'index': len(debug_results_list),
                            'number': result_num,
                            'path': temp_candidate_path,
                            'score': candidate_mask.get('predicted_iou', 0) * 10,
                            'iteration': iteration,
                            'params': current_iteration_params.copy(),
                            'approved': False,
                            'wrong_selection': False,
                            'is_inverted': False,
                            'is_candidate': True,
                            'candidate_index': idx,
                            'area': candidate_mask['area'],
                            'predicted_iou': candidate_mask.get('predicted_iou', 0),
                            'stability_score': candidate_mask.get('stability_score', 0)
                        })
                        
                        # Inverted mask
                        if mask_cand.dtype != bool:
                            mask_bool_cand = mask_cand.astype(bool)
                        else:
                            mask_bool_cand = mask_cand
                        inverted_mask_cand = ~mask_bool_cand
                        mask_array_inv_cand = (inverted_mask_cand.astype(np.uint8) * 255).astype(np.uint8)
                        image_array_inv_cand = np.array(image_pil.copy())
                        image_array_inv_cand[:, :, 3] = mask_array_inv_cand
                        result_img_inv_cand = Image.fromarray(image_array_inv_cand, 'RGBA')
                        temp_candidate_path_inv = f"{output_path}.candidate_{iteration}_{idx}_inv.png"
                        result_img_inv_cand.save(temp_candidate_path_inv, "PNG")
                        
                        result_num = len(debug_results_list) + 1
                        debug_results_list.append({
                            'index': len(debug_results_list),
                            'number': result_num,
                            'path': temp_candidate_path_inv,
                            'score': candidate_mask.get('predicted_iou', 0) * 10,
                            'iteration': iteration,
                            'params': current_iteration_params.copy(),
                            'approved': False,
                            'wrong_selection': False,
                            'is_inverted': True,
                            'is_candidate': True,
                            'candidate_index': idx,
                            'area': image_rgb.shape[0] * image_rgb.shape[1] - candidate_mask['area'],
                            'predicted_iou': candidate_mask.get('predicted_iou', 0),
                            'stability_score': candidate_mask.get('stability_score', 0)
                        })
                    
                    log_debug(f"Added {len(candidate_masks) * 2} candidates to debug results list (no LLM mode)")
                
                # Create result from selected mask
                mask_array = (mask * 255).astype(np.uint8)
                
                # Apply mask to alpha channel
                image_array = np.array(image_pil)
                image_array[:, :, 3] = mask_array
                
                result = Image.fromarray(image_array, 'RGBA')
                
                # Save temporary result for evaluation
                temp_output = output_path if iteration == max_iterations - 1 else f"{output_path}.temp_{iteration}.png"
                result.save(temp_output, "PNG")
            
            # 2. Check result with LLM censor
            if llm_censor and llm_censor.enabled:
                try:
                    log_debug(f"LLM censor: evaluating result of iteration {iteration + 1}...")
                    print(f"[SAM] LLM evaluation iteration {iteration + 1} (timeout: 5 min, can interrupt with Ctrl+C)...", file=sys.stderr)
                    # Note: Timeout is handled inside llm_censor (requests timeout=300 = 5 minutes)
                    # If LLM hangs, it will raise an exception which we catch
                    try:
                        evaluation = llm_censor.evaluate_segmentation(image_path, temp_output)
                        print(f"[SAM] LLM evaluation completed", file=sys.stderr)
                    except Exception as e:
                        log_error(f"LLM evaluation failed or timed out: {e}. Using current result without LLM approval.")
                        print(f"[WARNING] LLM evaluation failed/timed out: {e}. Continuing without LLM approval.", file=sys.stderr)
                        # Continue without LLM approval - use current result
                        evaluation = {'approved': False, 'quality_score': 5, 'wrong_selection': False}
                    
                    quality_score = evaluation.get('quality_score', 5)
                    approved = evaluation.get('approved', False)
                    wrong_selection = evaluation.get('wrong_selection', False)
                    
                    log_debug(f"LLM evaluation: approved={approved}, score={quality_score}, wrong_selection={wrong_selection}, issues={evaluation.get('issues', [])}")
                    
                    # Track if we've already saved results in debug mode (to avoid duplicates)
                    results_saved_in_debug = False
                    
                    # In debug mode, always try inverted mask to give user more voting options
                    # (even if wrong_selection=False, inverted might still be better or different)
                    # If wrong selection detected (background kept instead of object), try inverting the mask
                    if wrong_selection or llm_debug:
                        log_debug("LLM detected wrong selection (background kept instead of object), trying inverted mask...")
                        
                        # Try inverted mask
                        # Ensure mask is boolean before inverting
                        if mask.dtype != bool:
                            mask_bool = mask.astype(bool)
                        else:
                            mask_bool = mask
                        inverted_mask = ~mask_bool
                        mask_array_inv = (inverted_mask.astype(np.uint8) * 255).astype(np.uint8)
                        
                        image_pil_inv = Image.fromarray(image_rgb).convert("RGBA")
                        image_array_inv = np.array(image_pil_inv)
                        image_array_inv[:, :, 3] = mask_array_inv
                        
                        result_inv = Image.fromarray(image_array_inv, 'RGBA')
                        temp_output_inv = f"{output_path}.inverted_{iteration}.png"
                        result_inv.save(temp_output_inv, "PNG")
                        
                        # Re-evaluate inverted result
                        try:
                            evaluation_inv = llm_censor.evaluate_segmentation(image_path, temp_output_inv)
                            approved_inv = evaluation_inv.get('approved', False)
                            wrong_selection_inv = evaluation_inv.get('wrong_selection', False)
                            quality_score_inv = evaluation_inv.get('quality_score', 0)
                            
                            log_debug(f"Inverted mask evaluation: approved={approved_inv}, wrong_selection={wrong_selection_inv}, score={quality_score_inv}")
                            
                            # In debug mode, always save both original and inverted as separate options
                            if llm_debug:
                                # Save original result first (before potential replacement)
                                result_num_orig = len(debug_results_list) + 1
                                result_path_debug_orig = f"{output_path}.debug_{result_num_orig:03d}.png"
                                result.save(result_path_debug_orig, "PNG")
                                debug_results_list.append({
                                    'index': len(debug_results_list),
                                    'number': result_num_orig,
                                    'path': result_path_debug_orig,
                                    'score': quality_score,
                                    'iteration': iteration,
                                    'params': current_iteration_params.copy(),
                                    'approved': approved,
                                    'wrong_selection': wrong_selection,
                                    'is_inverted': False
                                })
                                log_debug(f"Saved debug result #{result_num_orig} (original) with score {quality_score}")
                                
                                # Save inverted result as separate option
                                result_num_inv = len(debug_results_list) + 1
                                result_path_debug_inv = f"{output_path}.debug_{result_num_inv:03d}.png"
                                result_inv.save(result_path_debug_inv, "PNG")
                                debug_results_list.append({
                                    'index': len(debug_results_list),
                                    'number': result_num_inv,
                                    'path': result_path_debug_inv,
                                    'score': quality_score_inv,
                                    'iteration': iteration,
                                    'params': current_iteration_params.copy(),
                                    'approved': approved_inv,
                                    'wrong_selection': wrong_selection_inv,
                                    'is_inverted': True
                                })
                                log_debug(f"Saved debug result #{result_num_inv} (inverted) with score {quality_score_inv}")
                                results_saved_in_debug = True
                                
                                # Use the better one for continuation
                                if not wrong_selection_inv and quality_score_inv > quality_score:
                                    log_debug("Inverted mask is better, using it for continuation")
                                    result = result_inv
                                    temp_output = temp_output_inv
                                    approved = approved_inv
                                    quality_score = quality_score_inv
                                    wrong_selection = False
                                elif wrong_selection:
                                    # Original was wrong, but inverted might be better
                                    if not wrong_selection_inv:
                                        log_debug("Original was wrong, using inverted for continuation")
                                        result = result_inv
                                        temp_output = temp_output_inv
                                        approved = approved_inv
                                        quality_score = quality_score_inv
                                        wrong_selection = False
                                    else:
                                        log_debug("Both original and inverted masks rejected")
                                        approved = False
                                        quality_score = 0
                            else:
                                # Normal mode: use better one
                                if not wrong_selection_inv and quality_score_inv > quality_score:
                                    log_debug("Inverted mask is better, using it")
                                    result = result_inv
                                    temp_output = temp_output_inv
                                    approved = approved_inv
                                    quality_score = quality_score_inv
                                    wrong_selection = False
                                    
                                    # Update best result tracking
                                    if quality_score > best_score:
                                        best_score = quality_score
                                        best_result = result.copy()
                                        # Save to numbered file for comparison
                                        result_path_with_score = f"{output_path}.score_{quality_score:.1f}_iter{iteration}_inv.png"
                                        best_result.save(result_path_with_score, "PNG")
                                        all_results.append((best_result.copy(), quality_score, result_path_with_score, iteration))
                                        log_debug(f"New best result (inverted, score: {quality_score})")
                                else:
                                    # Inverted mask is also wrong, reject both
                                    log_debug("Inverted mask is also wrong, rejecting")
                                    # In debug mode, keep all files
                                    if not llm_debug:
                                        try:
                                            os.remove(temp_output_inv)
                                        except:
                                            pass
                                    approved = False
                                    quality_score = 0
                        except Exception as e:
                            log_debug(f"Error evaluating inverted mask: {e}")
                            approved = False
                            quality_score = 0
                        
                        if not approved:
                            log_debug("Both original and inverted masks rejected")
                    
                    # Save all results (not just "best") for final comparison
                    # IMPORTANT: Don't overwrite output_path until we know which is truly best
                    # Skip if already saved in debug mode (when inverted mask was evaluated)
                    if not wrong_selection and not results_saved_in_debug:
                        # In debug mode, save with sequential number, otherwise with score
                        if llm_debug:
                            result_num = len(debug_results_list) + 1
                            result_path_debug = f"{output_path}.debug_{result_num:03d}.png"
                            result.save(result_path_debug, "PNG")
                            all_results.append((result.copy(), quality_score, result_path_debug, iteration))
                            # Store full debug info for later voting
                            debug_results_list.append({
                                'index': len(debug_results_list),
                                'number': result_num,
                                'path': result_path_debug,
                                'score': quality_score,
                                'iteration': iteration,
                                'params': current_iteration_params.copy(),
                                'approved': approved,
                                'wrong_selection': wrong_selection,
                                'is_inverted': False
                            })
                            log_debug(f"Saved debug result #{result_num} with score {quality_score} to {result_path_debug}")
                        else:
                            result_path_with_score = f"{output_path}.score_{quality_score:.1f}_iter{iteration}.png"
                            result.save(result_path_with_score, "PNG")
                            all_results.append((result.copy(), quality_score, result_path_with_score, iteration))
                            log_debug(f"Saved result with score {quality_score} to {result_path_with_score}")
                        
                        # Update best result tracking (but don't overwrite final output yet)
                        if quality_score > best_score:
                            best_score = quality_score
                            best_result = result.copy()
                            log_debug(f"New best result so far (score: {quality_score})")
                    
                    # Continue with parameter adjustment and comparison (debug mode still uses LLM suggestions)
                    # If result is approved, we can use it, but still compare with all results at the end
                    if approved and not wrong_selection:
                        log_debug(f"LLM censor approved result at iteration {iteration + 1}")
                        
                        # Don't save parameters here - they will be saved only when user selects in voting window
                        # This ensures parameters are saved as a preset for future use, not per-image
                        
                        # Don't return immediately - continue to compare with all results
                        # But mark this as approved for final selection
                        log_debug(f"Result approved, but will compare with all results at the end")
                    
                    # If not last iteration, adjust parameters
                    if iteration < max_iterations - 1:
                        # If wrong selection detected, make more aggressive parameter changes
                        if wrong_selection:
                            log_debug("Wrong selection detected, making aggressive parameter adjustments")
                            # Try different approach - increase points, adjust thresholds
                            sam_params = {
                                'points_per_side': min(64, sam_params.get('points_per_side', 32) + 8),
                                'pred_iou_thresh': max(0.5, sam_params.get('pred_iou_thresh', 0.88) - 0.1),
                                'stability_score_thresh': max(0.5, sam_params.get('stability_score_thresh', 0.95) - 0.05),
                                'min_mask_region_area': max(0, sam_params.get('min_mask_region_area', 100) - 50),
                                'crop_n_layers': min(3, sam_params.get('crop_n_layers', 1) + 1),
                                'crop_n_points_downscale_factor': max(1, sam_params.get('crop_n_points_downscale_factor', 2) - 1),
                                'box_nms_thresh': max(0.1, sam_params.get('box_nms_thresh', 0.7) - 0.1),
                            }
                            log_debug(f"Aggressive parameter adjustment: {sam_params}")
                        else:
                            new_params = llm_censor.suggest_sam_parameters(evaluation, sam_params)
                            if new_params:
                                log_debug(f"LLM suggested new parameters: {new_params}")
                                sam_params = new_params
                        
                        # Don't delete temp file if it's already saved in all_results
                        # Keep all results for final comparison
                        # In debug mode, never delete any files
                        if not llm_debug:
                            saved_paths = {path for _, _, path, _ in all_results}
                            try:
                                if temp_output != output_path and temp_output not in saved_paths:
                                    # This temp file is not in our saved results, safe to delete
                                    if os.path.exists(temp_output):
                                        os.remove(temp_output)
                                        log_debug(f"Removed unused temp file: {temp_output}")
                            except:
                                pass
                        continue
                    
                except Exception as e:
                    log_error(f"Error evaluating with LLM: {e}. Saving current result for comparison.")
                    # In case of LLM error, save result for comparison but continue
                    if result is not None:
                        if llm_debug:
                            # Save to debug_results_list
                            result_num = len(debug_results_list) + 1
                            result_path_debug = f"{output_path}.debug_{result_num:03d}.png"
                            result.save(result_path_debug, "PNG")
                            debug_results_list.append({
                                'index': len(debug_results_list),
                                'number': result_num,
                                'path': result_path_debug,
                                'score': 4.0,  # Lower score for error case
                                'iteration': iteration,
                                'params': current_iteration_params.copy(),
                                'approved': False,
                                'wrong_selection': False
                            })
                            log_debug(f"Saved debug result #{result_num} from error case to {result_path_debug}")
                        else:
                            # Save to all_results for final comparison
                            result_path_with_error = f"{output_path}.error_iter{iteration}.png"
                            result.save(result_path_with_error, "PNG")
                            all_results.append((result.copy(), 4.0, result_path_with_error, iteration))
                            log_debug(f"Saved result from error case (score: 4.0) to {result_path_with_error}")
                        # Don't return - continue to compare with all results at the end
                    # If result is None, we need to create it from the last mask
                    log_debug("Result is None, creating from last processed mask")
                    if masks:
                        # Use the best mask we have
                        best_mask = sorted_masks[0]
                        mask = best_mask['segmentation']
                        image_pil = Image.fromarray(image_rgb).convert("RGBA")
                        mask_array = (mask * 255).astype(np.uint8)
                        image_array = np.array(image_pil)
                        image_array[:, :, 3] = mask_array
                        result = Image.fromarray(image_array, 'RGBA')
                        result.save(output_path, "PNG")
                        return result
                    else:
                        raise ValueError("No masks generated and LLM evaluation failed")
            else:
                # LLM censor disabled - save result
                log_debug(f"LLM censor disabled - processing iteration {iteration + 1}")
                print(f"[SAM] LLM censor disabled - processing iteration {iteration + 1}", file=sys.stderr)
                
                # In debug mode, save numbered result and continue
                if llm_debug:
                    result_num = len(debug_results_list) + 1
                    result_path_debug = f"{output_path}.debug_{result_num:03d}.png"
                    result.save(result_path_debug, "PNG")
                    debug_results_list.append({
                        'index': len(debug_results_list),
                        'number': result_num,
                        'path': result_path_debug,
                        'score': 5.0,  # Default score
                        'iteration': iteration,
                        'params': current_iteration_params.copy(),
                        'approved': False,
                        'wrong_selection': False
                    })
                    log_debug(f"Saved debug result #{result_num} to {result_path_debug}")
                    # Continue to collect all iterations in debug mode
                    if iteration < max_iterations - 1:
                        continue
                    else:
                        # Last iteration - return debug results for voting
                        return debug_results_list
                
                # Normal mode: save result to final output and return immediately
                # Without LLM, we don't need multiple iterations - just use first result
                if result is not None:
                    result.save(output_path, "PNG")
                    log_debug(f"Saved result to {output_path}")
                    print(f"[SAM] Saved result to {os.path.basename(output_path)}", file=sys.stderr)
                    return result
                else:
                    # Result should have been created above, but if not, raise error
                    raise ValueError("Result is None but LLM censor is disabled")
        
        # If reached end of iterations, compare ALL results and choose the truly best one
        # IMPORTANT: Compare all saved results, not just the last "best" one
        # Also check for any saved result files in case process was interrupted
        if not all_results:
            log_debug("No results in all_results, checking for saved result files...")
            import glob
            import re
            base_path = str(output_path)
            # Look for files with score pattern: *.score_*.png
            saved_files = glob.glob(f"{base_path}.score_*.png")
            if saved_files:
                log_debug(f"Found {len(saved_files)} saved result files, loading them...")
                for saved_file in saved_files:
                    try:
                        # Extract score from filename
                        match = re.search(r'score_([\d.]+)_iter(\d+)', saved_file)
                        if match:
                            score = float(match.group(1))
                            iter_num = int(match.group(2))
                            saved_result = Image.open(saved_file).convert("RGBA")
                            all_results.append((saved_result, score, saved_file, iter_num))
                            log_debug(f"Loaded saved result: score={score}, iteration={iter_num}, file={os.path.basename(saved_file)}")
                    except Exception as e:
                        log_debug(f"Error loading saved file {saved_file}: {e}")
        
        # Check for debug results first
        if llm_debug and debug_results_list:
            # Debug mode: return debug info for voting
            # Save first result as placeholder
            first_result_path = debug_results_list[0]['path']
            import shutil
            shutil.copy2(first_result_path, output_path)
            log_debug(f"Debug mode: {len(debug_results_list)} results saved, waiting for user vote")
            print(f"[DEBUG] {len(debug_results_list)} results saved and numbered. Voting window will appear.", file=sys.stderr)
            return debug_results_list  # Return debug info instead of image
        
        if all_results:
            
            # Normal mode: sort and select best
            all_results.sort(key=lambda x: x[1], reverse=True)
            best_final_result, best_final_score, best_final_path, best_final_iter = all_results[0]
            
            log_debug(f"Comparing {len(all_results)} results:")
            for idx, (res, score, path, iter_num) in enumerate(all_results[:5]):  # Show top 5
                log_debug(f"  [{idx+1}] Score: {score:.1f}, Iteration: {iter_num}, Path: {os.path.basename(path)}")
            
            # Save the truly best result to final output
            best_final_result.save(output_path, "PNG")
            log_debug(f"Selected best result (score: {best_final_score:.1f}, iteration: {best_final_iter}) as final output")
            
            # Keep the best result file, but optionally clean up others
            # Don't delete immediately - user might want to compare
            log_debug(f"All results saved with scores. Best: {os.path.basename(best_final_path)}")
            log_debug(f"Other results available for comparison: {len(all_results)-1} files")
            
            # Print summary to console for user
            print(f"\n[INFO] Processed {len(all_results)} result(s). Best score: {best_final_score:.1f}", file=sys.stderr)
            if len(all_results) > 1:
                print(f"[INFO] All results saved with scores in filename. You can compare them manually.", file=sys.stderr)
                print(f"[INFO] Best result saved to: {os.path.basename(output_path)}", file=sys.stderr)
                print(f"[INFO] Other results: {', '.join([os.path.basename(path) for _, _, path, _ in all_results[1:5]])}", file=sys.stderr)
            
            return best_final_result
        elif best_result is not None:
            # Fallback: use tracked best result
            best_result.save(output_path, "PNG")
            log_debug(f"Saved tracked best result (score: {best_score}) to {output_path}")
            return best_result
        else:
            # No results in all_results, use last result
            if result is not None:
                log_debug(f"Saving final result to: {output_path}")
                result.save(output_path, "PNG")
                log_debug(f"Successfully processed image with SAM: {result.size}")
                return result
            else:
                # Last resort: try to find any saved temp file
                log_debug("No result available, checking for temp files...")
                for i in range(max_iterations):
                    temp_file = f"{output_path}.temp_{i}.png"
                    if os.path.exists(temp_file):
                        log_debug(f"Found temp file: {temp_file}, using it as result")
                        result = Image.open(temp_file).convert("RGBA")
                        result.save(output_path, "PNG")
                        return result
                raise ValueError("No result generated and no temp files found")
        
    except KeyboardInterrupt:
        # User interrupted - try to save best result if available
        log_error("Processing interrupted by user")
        print(f"[SAM] Process interrupted. Looking for saved results...", file=sys.stderr)
        
        # First, try to load any saved result files
        if not all_results:
            import glob
            import re
            base_path = str(output_path)
            saved_files = glob.glob(f"{base_path}.score_*.png")
            if saved_files:
                log_debug(f"Found {len(saved_files)} saved result files after interruption")
                for saved_file in saved_files:
                    try:
                        match = re.search(r'score_([\d.]+)_iter(\d+)', saved_file)
                        if match:
                            score = float(match.group(1))
                            iter_num = int(match.group(2))
                            saved_result = Image.open(saved_file).convert("RGBA")
                            all_results.append((saved_result, score, saved_file, iter_num))
                    except Exception as e:
                        log_debug(f"Error loading saved file {saved_file}: {e}")
        
        # Use best from all_results if available
        if all_results:
            all_results.sort(key=lambda x: x[1], reverse=True)
            best_final_result, best_final_score, best_final_path, _ = all_results[0]
            best_final_result.save(output_path, "PNG")
            print(f"[SAM] Saved best available result (score: {best_final_score:.1f}) to {os.path.basename(output_path)}", file=sys.stderr)
            log_debug(f"Saved best available result (score: {best_final_score:.1f}) from interruption")
            return best_final_result
        
        # Fallback to best_result
        if best_result is not None and os.path.exists(output_path):
            log_debug(f"Best result already saved to {output_path}")
            return best_result
        elif best_result is not None:
            best_result.save(output_path, "PNG")
            log_debug(f"Saved best result to {output_path} before interruption")
            return best_result
        else:
            # Try to find any temp file
            for i in range(max_iterations):
                temp_file = f"{output_path}.temp_{i}.png"
                if os.path.exists(temp_file):
                    result = Image.open(temp_file).convert("RGBA")
                    result.save(output_path, "PNG")
                    log_debug(f"Saved temp file {temp_file} to {output_path} before interruption")
                    return result
            raise KeyboardInterrupt("Processing interrupted and no results to save")
    except Exception as e:
        error_msg = f"Error removing background with SAM: {e}"
        log_error(error_msg, sys.exc_info())
        
        # Try to save best result even on error
        if best_result is not None:
            try:
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    best_result.save(output_path, "PNG")
                    log_debug(f"Saved best result to {output_path} despite error")
            except:
                pass
        
        raise Exception(error_msg)


def normalize_method_name(method):
    """Normalize legacy method names."""
    method = method.lower()
    legacy_map = {
        'sam': 'sam_cpu'
    }
    return legacy_map.get(method, method)


def remove_background(image_path, output_path=None, method='rembg_cpu', llm_censor=None, llm_max_iterations=3, save_successful_params=False, llm_debug=False):
    """
    Unified function to remove background using specified method.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output (if None, creates _nobg.png suffix)
        method (str): Method to use ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
        llm_censor: LLM censor instance (only used for SAM methods)
        llm_max_iterations: Maximum iterations for LLM parameter tuning (default: 3)
        save_successful_params: Whether to save successful parameters for next time (default: False)
        llm_debug: Debug mode - keep all intermediate results and return debug info (default: False)
    
    Returns:
        PIL.Image: Image with background removed, or dict with debug info if llm_debug=True, or None on error
    """
    method = normalize_method_name(method)
    log_debug(f"remove_background called with method: {method}, image: {image_path}, llm_censor: {llm_censor is not None}, debug: {llm_debug}")
    
    try:
        if method == 'rembg_cpu':
            return remove_background_rembg_cpu(image_path, output_path)
        elif method == 'rembg_gpu':
            return remove_background_rembg_gpu(image_path, output_path)
        elif method == 'sam_cpu':
            return remove_background_sam(image_path, output_path, device='cpu', llm_censor=llm_censor, max_iterations=llm_max_iterations, save_successful_params=save_successful_params, llm_debug=llm_debug)
        elif method == 'sam_gpu':
            return remove_background_sam(image_path, output_path, device='cuda', llm_censor=llm_censor, max_iterations=llm_max_iterations, save_successful_params=save_successful_params, llm_debug=llm_debug)
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