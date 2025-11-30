"""
Batch sticker creation script.

Usage:
    python batch_create_sticker.py file1.png file2.png ... [--parallel] [--width 20]
"""

import sys
import os
from pathlib import Path

# Setup local environment first - MUST be imported before any other imports
from local_env import LOCAL_SITE_PACKAGES
PROJECT_ROOT = Path(__file__).parent.absolute()

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import create_sticker function
try:
    from create_sticker import create_sticker
    CREATE_STICKER_AVAILABLE = True
except ImportError:
    CREATE_STICKER_AVAILABLE = False
    print("Error: create_sticker module not found.")
    print("Make sure create_sticker.py is in the same directory.")
    sys.exit(1)


def create_sticker_single(image_path, outline_width=20, bg_method='rembg_cpu'):
    """
    Creates a sticker from a single image.
    
    Args:
        image_path (str): Path to source image
        outline_width (int): White outline width in pixels
        bg_method (str): Background removal method
    
    Returns:
        tuple: (success: bool, input_path: str, output_path: str, error: str or None)
    """
    try:
        input_path = Path(image_path)
        
        if not input_path.exists():
            return (False, str(input_path), None, f"File not found: {input_path}")
        
        # Create output filename
        output_path = input_path.parent / f"{input_path.stem}_sticker{input_path.suffix}"
        
        # Create sticker
        create_sticker(
            str(input_path),
            str(output_path),
            outline_width=outline_width,
            smooth=True,
            auto_remove_bg=True,
            use_ollama=False,
            bg_method=bg_method
        )
        
        return (True, str(input_path), str(output_path), None)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Failed to create sticker from {image_path} with {bg_method}:", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        return (False, str(image_path), None, str(e))


def process_sequential(image_files, outline_width=20, bg_method='rembg_cpu'):
    """
    Process images sequentially (one after another).
    
    Args:
        image_files (list): List of image file paths
        outline_width (int): White outline width in pixels
        bg_method (str): Background removal method
    
    Returns:
        dict: Statistics about processing
    """
    total = len(image_files)
    success_count = 0
    error_count = 0
    
    print(f"Processing {total} images sequentially...")
    print(f"Background removal method: {bg_method}")
    print("=" * 60)
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total}] Processing: {os.path.basename(image_file)}")
        
        success, input_path, output_path, error = create_sticker_single(image_file, outline_width, bg_method)
        
        if success:
            success_count += 1
            print(f"  [OK] Success: {os.path.basename(output_path)}")
        else:
            error_count += 1
            print(f"  [ERROR] Error: {error}")
    
    return {
        'total': total,
        'success': success_count,
        'errors': error_count
    }


def process_parallel(image_files, outline_width=20, max_workers=None, bg_method='rembg_cpu'):
    """
    Process images in parallel using multiple threads.
    
    Args:
        image_files (list): List of image file paths
        outline_width (int): White outline width in pixels
        max_workers (int): Maximum number of worker threads (None = auto-detect)
        bg_method (str): Background removal method
    
    Returns:
        dict: Statistics about processing
    """
    total = len(image_files)
    
    # Determine number of workers (all cores except one)
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, cpu_count - 1)  # Leave one core free
    
    print(f"Processing {total} images in parallel...")
    print(f"Background removal method: {bg_method}")
    print(f"Using {max_workers} worker threads (CPU cores: {multiprocessing.cpu_count()})")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(create_sticker_single, img, outline_width, bg_method): img 
            for img in image_files
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            image_file = future_to_file[future]
            
            try:
                success, input_path, output_path, error = future.result()
                
                if success:
                    success_count += 1
                    print(f"[{completed}/{total}] [OK] {os.path.basename(output_path)}")
                else:
                    error_count += 1
                    print(f"[{completed}/{total}] [ERROR] {os.path.basename(image_file)}: {error}")
                    
            except Exception as e:
                error_count += 1
                print(f"[{completed}/{total}] [ERROR] {os.path.basename(image_file)}: {e}")
    
    return {
        'total': total,
        'success': success_count,
        'errors': error_count
    }


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Batch create stickers from multiple images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_create_sticker.py image1.png image2.png image3.png
  python batch_create_sticker.py *.png --parallel
  python batch_create_sticker.py image1.png image2.png --parallel --width 15
  python batch_create_sticker.py *.png --parallel --workers 4
        """
    )
    
    parser.add_argument('images', nargs='+', help='Image files to process')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Process images in parallel (uses all CPU cores except one)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker threads for parallel processing (default: CPU cores - 1)')
    parser.add_argument('--width', type=int, default=20,
                       help='White outline width in pixels (default: 20)')
    parser.add_argument('--method', '-m', default='rembg_cpu',
                       choices=['rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu', 'sam'],
                       help='Background removal method (default: rembg_cpu)')
    
    args = parser.parse_args()
    
    # Check if create_sticker is available
    if not CREATE_STICKER_AVAILABLE:
        print("Error: create_sticker module not found.")
        print("Make sure create_sticker.py is in the same directory.")
        sys.exit(1)
    
    # Validate input files
    valid_files = []
    for img_path in args.images:
        path = Path(img_path)
        if path.exists() and path.is_file():
            # Check if it's an image file
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                valid_files.append(str(path))
            else:
                print(f"Warning: Skipping non-image file: {img_path}")
        else:
            print(f"Warning: File not found: {img_path}")
    
    if not valid_files:
        print("Error: No valid image files found.")
        sys.exit(1)
    
    print(f"\nFound {len(valid_files)} valid image file(s)")
    print("=" * 60)
    
    if args.method == 'sam':
        print("Note: 'sam' has been renamed to 'sam_cpu'.")
        args.method = 'sam_cpu'

    # Process images
    if args.parallel:
        stats = process_parallel(valid_files, outline_width=args.width, max_workers=args.workers, bg_method=args.method)
    else:
        stats = process_sequential(valid_files, outline_width=args.width, bg_method=args.method)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 60)
    
    if stats['errors'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

