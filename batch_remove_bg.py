"""
Batch background removal script.

Usage:
    python batch_remove_bg.py file1.png file2.png ... [--parallel]
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
        print("Error: Background removal libraries not installed.")
        print("Install with: pip install --target lib/site-packages rembg onnxruntime")
        sys.exit(1)


def remove_background_single(image_path, method='rembg_cpu'):
    """
    Removes background from a single image.
    
    Args:
        image_path (str): Path to source image
        method (str): Method to use ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
    
    Returns:
        tuple: (success: bool, input_path: str, output_path: str, error: str or None)
    """
    try:
        input_path = Path(image_path)
        
        if not input_path.exists():
            return (False, str(input_path), None, f"File not found: {input_path}")
        
        # Create output filename
        output_path = input_path.parent / f"{input_path.stem}_nobg{input_path.suffix}"
        
        # Use unified background removal
        if BG_REMOVAL_AVAILABLE:
            remove_background(str(input_path), str(output_path), method=method)
        else:
            # Fallback to old rembg method
            if not REMBG_AVAILABLE:
                return (False, str(image_path), None, "Background removal libraries not available")
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
            
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
        
        return (True, str(input_path), str(output_path), None)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] Failed to process {image_path} with {method}:", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        return (False, str(image_path), None, str(e))


def process_sequential(image_files, method='rembg_cpu'):
    """
    Process images sequentially (one after another).
    
    Args:
        image_files (list): List of image file paths
        method (str): Background removal method
    
    Returns:
        dict: Statistics about processing
    """
    total = len(image_files)
    success_count = 0
    error_count = 0
    
    print(f"Processing {total} images sequentially...")
    print(f"Method: {method}")
    print("=" * 60)
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total}] Processing: {os.path.basename(image_file)}")
        
        success, input_path, output_path, error = remove_background_single(image_file, method=method)
        
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


def process_parallel(image_files, max_workers=None, method='rembg_cpu'):
    """
    Process images in parallel using multiple threads.
    
    Args:
        image_files (list): List of image file paths
        max_workers (int): Maximum number of worker threads (None = auto-detect)
        method (str): Background removal method
    
    Returns:
        dict: Statistics about processing
    """
    total = len(image_files)
    
    # Determine number of workers (all cores except one)
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, cpu_count - 1)  # Leave one core free
    
    print(f"Processing {total} images in parallel...")
    print(f"Method: {method}")
    print(f"Using {max_workers} worker threads (CPU cores: {multiprocessing.cpu_count()})")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(remove_background_single, img, method): img 
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
        description='Batch remove background from multiple images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_remove_bg.py image1.png image2.png image3.png
  python batch_remove_bg.py *.png --parallel
  python batch_remove_bg.py image1.png image2.png --parallel --workers 4
        """
    )
    
    parser.add_argument('images', nargs='+', help='Image files to process')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Process images in parallel (uses all CPU cores except one)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker threads for parallel processing (default: CPU cores - 1)')
    parser.add_argument('--method', '-m', default='rembg_cpu',
                       choices=['rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu', 'sam'],
                       help='Background removal method (default: rembg_cpu)')
    
    args = parser.parse_args()
    
    # Check if background removal is available
    if not BG_REMOVAL_AVAILABLE and not REMBG_AVAILABLE:
        print("Error: Background removal libraries are not installed.")
        print("Install with: pip install rembg onnxruntime")
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
    
    # Normalize legacy alias
    if args.method == 'sam':
        print("Note: 'sam' method has been renamed to 'sam_cpu'.")
        args.method = 'sam_cpu'

    # Process images
    if args.parallel:
        stats = process_parallel(valid_files, max_workers=args.workers, method=args.method)
    else:
        stats = process_sequential(valid_files, method=args.method)
    
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

