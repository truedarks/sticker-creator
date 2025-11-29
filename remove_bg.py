"""
Script for removing background from images.

Usage:
    python remove_bg.py input.png [output.png]
    
Or as a module:
    from remove_bg import remove_background_only
    remove_background_only('input.png', 'output.png')
"""

import sys
import os
import subprocess
import argparse

# Import for background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def check_and_install_rembg():
    """
    Checks if rembg is installed and prompts user to install it if not.
    Allows user to choose installation location.
    
    Returns:
        bool: True if rembg is available, False otherwise
    """
    global REMBG_AVAILABLE
    
    if REMBG_AVAILABLE:
        return True
    
    print("\n" + "="*60)
    print("WARNING: Background removal library not found")
    print("="*60)
    print("\nThe 'rembg' library will be downloaded for local use.")
    print("This library is required for background removal.")
    print("\nThe library will be installed using pip.")
    print("You can choose to install it:")
    print("  1. System-wide (default Python installation)")
    print("  2. User-only (current user only)")
    print("  3. Cancel installation")
    
    while True:
        choice = input("\nChoose installation type [1/2/3] (default: 1): ").strip()
        
        if choice == '' or choice == '1':
            # System-wide installation
            install_cmd = [sys.executable, '-m', 'pip', 'install', 'rembg', 'onnxruntime']
            break
        elif choice == '2':
            # User-only installation
            install_cmd = [sys.executable, '-m', 'pip', 'install', '--user', 'rembg', 'onnxruntime']
            break
        elif choice == '3':
            print("Installation cancelled. Background removal cannot proceed.")
            return False
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nInstalling rembg and dependencies...")
    print("This may take a few minutes on first run...")
    
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print("[OK] Installation completed successfully!")
        
        # Try to import again
        try:
            from rembg import remove
            REMBG_AVAILABLE = True
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


def remove_background_only(image_path, output_path=None):
    """
    Removes background from image without adding outline.
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save result (if None, creates _nobg.png suffix)
    
    Returns:
        PIL.Image: Image with background removed, or None on error
    """
    # Check and install rembg if needed
    if not REMBG_AVAILABLE:
        if not check_and_install_rembg():
            print("Error: rembg not available. Cannot remove background.")
            return None
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_nobg.png"
    
    try:
        print(f"Removing background from: {image_path}")
        
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
            output_data = remove(input_data)
        
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
        
        print(f"[OK] Background removed. Saved to: {output_path}")
        
        from PIL import Image
        return Image.open(output_path).convert("RGBA")
        
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
        """
    )
    
    parser.add_argument('input', help='Path to source image')
    parser.add_argument('output', nargs='?', default=None,
                       help='Path to save result (default: input_nobg.png)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    try:
        result = remove_background_only(args.input, args.output)
        if result is None:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

