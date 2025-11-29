"""
Script for creating stickers with white outline from images.

Usage:
    python create_sticker.py input.png output.png [--width 20]
    
Or as a module:
    from create_sticker import create_sticker
    create_sticker('input.png', 'output.png', outline_width=20)
"""

from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage
import argparse
import sys
import os
import subprocess

# Import for background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Import for Ollama (optional)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


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
    print("This library is required for automatic background removal.")
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
            print("Installation cancelled. Background removal will be skipped.")
            return False
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nInstalling rembg and dependencies...")
    print("This may take a few minutes on first run...")
    
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print("✓ Installation completed successfully!")
        
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


def needs_background_removal(image_path):
    """
    Checks if background removal is needed for the image.
    
    Returns:
        bool: True if background needs to be removed, False if already transparent
    """
    try:
        img = Image.open(image_path)
        
        # If image doesn't have alpha channel, definitely needs processing
        if img.mode not in ('RGBA', 'LA'):
            return True
        
        # Check if there are transparent pixels
        if img.mode == 'RGBA':
            alpha = np.array(img.split()[3])
            # If all pixels are opaque, there's a background
            if np.all(alpha == 255):
                return True
            # If there's at least some transparency, consider background already removed
            return False
        
        return True
    except Exception as e:
        print(f"Error checking image: {e}")
        return True


def remove_background_rembg(image_path, output_path=None):
    """
    Removes background from image using rembg.
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save (if None, overwrites source)
    
    Returns:
        PIL.Image: Image with background removed
    """
    if not REMBG_AVAILABLE:
        raise ImportError("rembg is not installed. Install with: pip install rembg")
    
    # Normalize and check input file path
    image_path = os.path.abspath(os.path.normpath(image_path))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")
    
    try:
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
            output_data = remove(input_data)
        
        # Save to temporary file if needed
        if output_path is None:
            output_path = image_path
        
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
        
        # Open and return as PIL Image
        result = Image.open(output_path).convert("RGBA")
        return result
    except Exception as e:
        print(f"Error removing background: {e}")
        raise


def analyze_with_ollama(image_path, model="llava"):
    """
    Analyzes image using Ollama LLM (optional).
    
    Args:
        image_path (str): Path to image
        model (str): Ollama model to use (default "llava")
    
    Returns:
        str: Image analysis or None if Ollama is unavailable
    """
    if not OLLAMA_AVAILABLE:
        return None
    
    try:
        import base64
        
        # Read image and encode to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Send request to Ollama (correct format for vision models)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": "Does this image have a background that should be removed? Answer yes or no only.",
            "images": [image_data],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        return None
    except requests.exceptions.ConnectionError:
        # Ollama is not running or unavailable
        return None
    except Exception as e:
        # If Ollama is unavailable, just ignore
        return None


def create_sticker(image_path, output_path, outline_width=20, smooth=True, auto_remove_bg=True, use_ollama=False):
    """
    Creates a sticker with white outline from an image.
    Automatically removes background if present.
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save result
        outline_width (int): White outline width in pixels (default 20)
        smooth (bool): Use outline smoothing (default True)
        auto_remove_bg (bool): Automatically remove background if present (default True)
        use_ollama (bool): Use Ollama for image analysis (default False)
    
    Returns:
        PIL.Image: Created sticker image
    """
    # Normalize and check input file path
    image_path = os.path.abspath(os.path.normpath(image_path))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")
    
    # Smart processing: remove background if needed
    processed_image_path = image_path
    temp_file = None
    
    if auto_remove_bg:
        # Check and install rembg if needed
        if not REMBG_AVAILABLE:
            if not check_and_install_rembg():
                print("Warning: rembg not available. Skipping background removal.")
                print("Install with: pip install rembg")
        
        if use_ollama and OLLAMA_AVAILABLE:
            print("Analyzing image with Ollama...")
            analysis = analyze_with_ollama(image_path)
            if analysis:
                print(f"Ollama analysis: {analysis}")
        
        if needs_background_removal(image_path):
            if not REMBG_AVAILABLE:
                print("Warning: rembg not installed. Skipping background removal.")
            else:
                print("Background detected. Removing background...")
                # Create temporary file for processed image
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.close()
                processed_image_path = temp_file.name
                
                try:
                    remove_background_rembg(image_path, processed_image_path)
                    print("✓ Background successfully removed")
                except Exception as e:
                    print(f"Error removing background: {e}")
                    processed_image_path = image_path
        else:
            print("Image already has transparent background. Skipping background removal.")
    
    # Open processed image
    try:
        original = Image.open(processed_image_path).convert("RGBA")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    finally:
        # Delete temporary file if it was created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    # Add padding for outline expansion outward
    padded_size = (original.width + outline_width * 2, original.height + outline_width * 2)
    padded_img = Image.new("RGBA", padded_size, (0, 0, 0, 0))
    padded_img.paste(original, (outline_width, outline_width), original)
    
    data = np.array(padded_img)
    
    # Get alpha channel (mask of opaque pixels)
    alpha = data[:, :, 3]
    mask = alpha > 0  # True for opaque pixels
    
    # Expand mask outward by outline_width pixels
    if smooth:
        # For smoother expansion, use multiple iterations with smaller kernel
        # This creates a smoother outline
        structure_size = 3
        structure = np.ones((structure_size, structure_size), dtype=bool)
        expanded_mask = mask.copy()
        for _ in range(outline_width):
            expanded_mask = ndimage.binary_dilation(expanded_mask, structure=structure, iterations=1)
    else:
        # Standard expansion
        structure_size = outline_width * 2 + 1
        structure = np.ones((structure_size, structure_size), dtype=bool)
        expanded_mask = ndimage.binary_dilation(mask, structure=structure, iterations=1)
    
    # Create mask only for outline (expanded mask minus original)
    outline_mask = expanded_mask & (~mask)
    
    # Create new image with fully transparent background
    result_data = np.zeros((padded_size[1], padded_size[0], 4), dtype=np.uint8)
    
    # Copy original image
    result_data[mask] = data[mask]
    
    # Fill outline area with white
    result_data[outline_mask] = [255, 255, 255, 255]
    
    # Create final image
    result = Image.fromarray(result_data, 'RGBA')
    
    # Apply smoothing for smoother outline edges
    if smooth:
        # Create mask for outline and edge areas
        result_data_array = np.array(result)
        alpha_result = result_data_array[:, :, 3]
        
        # Find outline edges for smoothing
        from scipy.ndimage import gaussian_filter
        
        # Apply light blur only to alpha channel for edge smoothing
        alpha_smooth = gaussian_filter(alpha_result.astype(np.float32), sigma=0.8)
        alpha_smooth = np.clip(alpha_smooth, 0, 255).astype(np.uint8)
        
        # Update alpha channel but preserve original colors
        result_data_array[:, :, 3] = alpha_smooth
        
        # Restore main image sharpness (don't blur it)
        result_data_array[mask] = data[mask]
        
        result = Image.fromarray(result_data_array, 'RGBA')
    
    # Crop to bounds (including outline)
    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)
    
    result.save(output_path, "PNG")
    
    print(f"✓ Sticker created: {output_path}")
    print(f"  White outline width: {outline_width} pixels")
    print(f"  Original size: {original.width}x{original.height}")
    print(f"  Sticker size: {result.width}x{result.height}")
    print(f"  Smoothing: {'enabled' if smooth else 'disabled'}")
    print(f"  Background: transparent")
    
    return result


def remove_background_only(image_path, output_path=None):
    """
    Removes background from image without adding outline.
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save result (if None, creates _nobg.png suffix)
    
    Returns:
        PIL.Image: Image with background removed
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
        result = remove_background_rembg(image_path, output_path)
        print(f"✓ Background removed. Saved to: {output_path}")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Creates sticker with white outline. Automatically removes background if present.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_sticker.py input.png output.png
  python create_sticker.py input.png output.png --width 30
  python create_sticker.py input.png output.png --width 20 --no-smooth
  python create_sticker.py input.png output.png --no-remove-bg
  python create_sticker.py input.png output.png --use-ollama
        """
    )
    
    parser.add_argument('input', help='Path to source image')
    parser.add_argument('output', help='Path to save result')
    parser.add_argument('--width', '-w', type=int, default=20,
                       help='White outline width in pixels (default: 20)')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable outline smoothing')
    parser.add_argument('--no-remove-bg', action='store_true',
                       help='Disable automatic background removal')
    parser.add_argument('--use-ollama', action='store_true',
                       help='Use Ollama for image analysis (requires running Ollama)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        print(f"Please check the file path and try again.", file=sys.stderr)
        sys.exit(1)
    
    # Check if input is actually a file
    if not os.path.isfile(args.input):
        print(f"Error: Path is not a file: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        create_sticker(
            args.input,
            args.output,
            outline_width=args.width,
            smooth=not args.no_smooth,
            auto_remove_bg=not args.no_remove_bg,
            use_ollama=args.use_ollama
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
