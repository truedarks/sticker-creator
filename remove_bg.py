"""
Script for removing background from images.

Usage:
    python remove_bg.py input.png [output.png] [--method rembg_cpu|rembg_gpu|sam_cpu|sam_gpu]
    
Or as a module:
    from remove_bg import remove_background_only
    remove_background_only('input.png', 'output.png', method='rembg_cpu')
"""

import sys
import os
from pathlib import Path

# Setup local environment first - MUST be imported before any other imports
from local_env import LOCAL_SITE_PACKAGES
PROJECT_ROOT = Path(__file__).parent.absolute()

import subprocess
import argparse

# Import unified background removal
try:
    from bg_removal import remove_background, get_available_methods
    BG_REMOVAL_AVAILABLE = True
except ImportError:
    BG_REMOVAL_AVAILABLE = False
    # Fallback to old rembg import
    try:
        from rembg import remove
        REMBG_AVAILABLE = True
    except ImportError:
        REMBG_AVAILABLE = False


def check_and_install_dependencies(method='rembg_cpu'):
    """
    Checks if required dependencies are installed and prompts user to install if not.
    
    Args:
        method (str): Method to check ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
    
    Returns:
        bool: True if dependencies are available, False otherwise
    """
    global BG_REMOVAL_AVAILABLE
    
    if BG_REMOVAL_AVAILABLE:
        available = get_available_methods()
        if method in available:
            return True
    
    print("\n" + "="*60)
    print("WARNING: Background removal library not found")
    print("="*60)
    
    normalized_method = method if method != 'sam' else 'sam_cpu'
    
    if normalized_method in ('sam_cpu', 'sam_gpu'):
        print("\nThe 'segment-anything' library is required for SAM.")
        packages = ['segment-anything', 'torch', 'torchvision', 'opencv-python']
        if normalized_method == 'sam_gpu':
            print("For GPU acceleration, install the CUDA-enabled PyTorch build.")
    elif method == 'rembg_gpu':
        print("\nThe 'rembg' and 'onnxruntime-gpu' libraries are required.")
        packages = ['rembg', 'onnxruntime-gpu']
    else:  # rembg_cpu
        print("\nThe 'rembg' library is required for background removal.")
        packages = ['rembg', 'onnxruntime']
    
    print("\nThe libraries will be installed using pip.")
    print("You can choose to install them:")
    print("  1. System-wide (default Python installation)")
    print("  2. User-only (current user only)")
    print("  3. Cancel installation")
    
    while True:
        choice = input("\nChoose installation type [1/2/3] (default: 1): ").strip()
        
        if choice == '' or choice == '1':
            # Install to local project directory
            LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
            
            # Uninstall onnxruntime (CPU) if installing GPU version
            if method == 'rembg_gpu':
                print("Checking for conflicting packages...")
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime'],
                    capture_output=True
                )
            
            install_cmd = [sys.executable, '-m', 'pip', 'install', '--target', str(LOCAL_SITE_PACKAGES)] + packages
            break
        elif choice == '2':
            # User-only installation (fallback)
            
            # Uninstall onnxruntime (CPU) if installing GPU version
            if method == 'rembg_gpu':
                print("Checking for conflicting packages...")
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime'],
                    capture_output=True
                )
            
            install_cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + packages
            break
        elif choice == '3':
            print("Installation cancelled. Background removal cannot proceed.")
            return False
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\nInstalling {', '.join(packages)}...")
    print("This may take a few minutes on first run...")
    
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print("[OK] Installation completed successfully!")
        
        # Try to import again
        try:
            from bg_removal import remove_background, get_available_methods
            BG_REMOVAL_AVAILABLE = True
            return True
        except ImportError:
            print("Warning: Installation completed but import failed.")
            print("You may need to restart the script.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def remove_background_only(image_path, output_path=None, method='rembg_cpu'):
    """
    Removes background from image without adding outline.
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save result (if None, creates _nobg.png suffix)
        method (str): Method to use ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
    
    Returns:
        PIL.Image: Image with background removed, or None on error
    """
    # Check and install dependencies if needed
    if not BG_REMOVAL_AVAILABLE:
        if not check_and_install_dependencies(method):
            print(f"Error: {method} not available. Cannot remove background.")
            return None
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_nobg.png"
    
    try:
        print(f"Removing background from: {image_path} (method: {method})")
        
        result = remove_background(image_path, output_path, method=method)
        
        print(f"[OK] Background removed. Saved to: {output_path}")
        return result
        
    except Exception as e:
        print(f"Error removing background: {e}")
        return None


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Removes background from image. Creates transparent background.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_bg.py input.png
  python remove_bg.py input.png output.png
  python remove_bg.py input.png --method rembg_gpu
  python remove_bg.py input.png --method sam_cpu
        """
    )
    
    parser.add_argument('input', help='Path to source image')
    parser.add_argument('output', nargs='?', default=None,
                       help='Path to save result (default: input_nobg.png)')
    parser.add_argument('--method', '-m', default='rembg_cpu',
                       choices=['rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu', 'sam'],
                       help='Background removal method (default: rembg_cpu)')
    
    args = parser.parse_args()
    
    if args.method == 'sam':
        print("Note: 'sam' has been renamed to 'sam_cpu'. Using CPU mode.\n")
        args.method = 'sam_cpu'
    
    if args.method in ('sam_cpu', 'sam_gpu'):
        print("SAM methods require: segment-anything, torch, torchvision, opencv-python")
        if args.method == 'sam_gpu':
            print("For GPU support install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/")
        print()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    try:
        result = remove_background_only(args.input, args.output, method=args.method)
        if result is None:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

