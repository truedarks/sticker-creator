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
import json

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

# Try to import LLM censor
try:
    from llm_censor import create_llm_censor
    LLM_CENSOR_AVAILABLE = True
except ImportError:
    LLM_CENSOR_AVAILABLE = False


def remove_background_single(image_path, method='rembg_cpu', llm_censor=None, llm_max_iterations=3, save_successful_params=False, llm_debug=False):
    """
    Removes background from a single image.
    
    Args:
        image_path (str): Path to source image
        method (str): Method to use ('rembg_cpu', 'rembg_gpu', 'sam_cpu', 'sam_gpu')
        llm_censor: LLM censor instance (only used for SAM methods)
        llm_max_iterations: Maximum iterations for LLM parameter tuning
        save_successful_params: Whether to save successful parameters for next time
        llm_debug: Debug mode - keep all intermediate results and show voting window
    
    Returns:
        tuple: (success: bool, input_path: str, output_path: str, error: str or None, debug_results: list or None)
    """
    input_path = Path(image_path)
    output_path = None
    
    try:
        if not input_path.exists():
            return (False, str(input_path), None, f"File not found: {input_path}")
        
        # Create output filename
        output_path = input_path.parent / f"{input_path.stem}_nobg{input_path.suffix}"
        
        # Use unified background removal
        if BG_REMOVAL_AVAILABLE:
            result = remove_background(str(input_path), str(output_path), method=method, 
                            llm_censor=llm_censor, llm_max_iterations=llm_max_iterations, save_successful_params=save_successful_params, llm_debug=llm_debug)
            
            # Check if result is debug_results (list) or image
            if result is not None and llm_debug and isinstance(result, list):
                # Debug mode returned list of results
                return (True, str(input_path), str(output_path), None, result)
            
            # For rembg methods or SAM without debug, result is PIL.Image or None
            # The file should already be saved by remove_background function
        else:
            # Fallback to old rembg method
            if not REMBG_AVAILABLE:
                return (False, str(image_path), None, "Background removal libraries not available")
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
            
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
        
        # Verify output file was created
        if output_path and output_path.exists() and output_path.stat().st_size > 0:
            return (True, str(input_path), str(output_path), None, None)
        else:
            return (False, str(input_path), str(output_path) if output_path else None, "Output file was not created or is empty", None)
        
    except KeyboardInterrupt:
        # User cancelled - don't treat as error
        raise
    except SystemExit:
        # System exit - re-raise
        raise
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}"
        
        # Check if output file was partially created
        if output_path and output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 0:
                # File exists but might be incomplete
                error_msg += f" (output file exists but may be incomplete: {file_size} bytes)"
        
        # Print detailed error to stderr
        print(f"[ERROR] Failed to process {image_path} with {method}:", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        
        # For critical errors, print full traceback
        if "access violation" in str(e).lower() or "0xc0000005" in str(e).lower():
            print("\n[CRITICAL] Access violation detected. Possible causes:", file=sys.stderr)
            print("  - GPU driver issues", file=sys.stderr)
            print("  - Library conflicts (onnxruntime, torch)", file=sys.stderr)
            print("  - Insufficient memory", file=sys.stderr)
            print("  - Corrupted CUDA installation", file=sys.stderr)
            print("\nTry:", file=sys.stderr)
            print("  - Use CPU method instead (rembg_cpu)", file=sys.stderr)
            print("  - Update NVIDIA drivers", file=sys.stderr)
            print("  - Restart the application", file=sys.stderr)
            print("\nFull traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        return (False, str(image_path), str(output_path) if output_path else None, error_msg, None)


def process_sequential(image_files, method='rembg_cpu', llm_censor=None, llm_max_iterations=3, save_successful_params=False, llm_debug=False):
    """
    Process images sequentially (one after another).
    
    Args:
        image_files (list): List of image file paths
        method (str): Background removal method
        llm_censor: LLM censor instance (only used for SAM methods)
        llm_max_iterations: Maximum iterations for LLM parameter tuning
        save_successful_params: Whether to save successful parameters for next time
        llm_debug: Debug mode - keep all intermediate results and show voting window
    
    Returns:
        dict: Statistics about processing
    """
    import time
    
    total = len(image_files)
    success_count = 0
    error_count = 0
    
    print(f"Processing {total} images sequentially...")
    print(f"Method: {method}")
    if method in ('sam_cpu', 'sam_gpu'):
        print(f"Note: SAM model will be loaded once and reused for all images.")
    print("=" * 60)
    
    total_start = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total}] Processing: {os.path.basename(image_file)}")
        
        start_time = time.time()
        try:
            result = remove_background_single(
                image_file, method=method, llm_censor=llm_censor, llm_max_iterations=llm_max_iterations, save_successful_params=save_successful_params, llm_debug=llm_debug
            )
            if len(result) == 5:
                success, input_path, output_path, error, debug_results = result
            else:
                # Backward compatibility
                success, input_path, output_path, error = result
                debug_results = None
        except KeyboardInterrupt:
            print(f"  [INTERRUPTED] Processing cancelled by user")
            # Check if output file was created before interruption
            input_path_obj = Path(image_file)
            output_path_check = input_path_obj.parent / f"{input_path_obj.stem}_nobg{input_path_obj.suffix}"
            if output_path_check.exists():
                print(f"  [INFO] Partial result saved to: {output_path_check}")
            raise
        elapsed = time.time() - start_time
        
        # Handle debug mode - save debug results to JSON file for GUI to pick up
        if llm_debug and debug_results and success:
            try:
                # Save debug results to JSON file for GUI to process
                debug_json_path = f"{output_path}.debug_results.json"
                debug_data = {
                    'original_path': str(input_path),
                    'output_path': str(output_path),
                    'results': debug_results
                }
                with open(debug_json_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_data, f, indent=2, ensure_ascii=False)
                print(f"  [DEBUG] Saved {len(debug_results)} debug results to {os.path.basename(debug_json_path)}")
                print(f"  [DEBUG] Voting window will appear in GUI")
            except Exception as e:
                import traceback
                print(f"  [WARNING] Could not save debug results: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        
        if success:
            success_count += 1
            print(f"  [OK] Success: {os.path.basename(output_path)} ({elapsed:.1f}s)")
        else:
            error_count += 1
            print(f"  [ERROR] Error: {error}")
    
    total_elapsed = time.time() - total_start
    
    return {
        'total': total,
        'success': success_count,
        'errors': error_count,
        'total_time': total_elapsed
    }


def process_parallel(image_files, max_workers=None, method='rembg_cpu', llm_censor=None, llm_max_iterations=3, llm_debug=False):
    """
    Process images in parallel using multiple threads.
    
    Args:
        image_files (list): List of image file paths
        max_workers (int): Maximum number of worker threads (None = auto-detect)
        method (str): Background removal method
        llm_censor: LLM censor instance (only used for SAM methods, note: not thread-safe for parallel processing)
        llm_max_iterations: Maximum iterations for LLM parameter tuning
        llm_debug: Debug mode - keep all intermediate results and show voting window
    
    Returns:
        dict: Statistics about processing
    """
    total = len(image_files)
    
    # For SAM methods, parallel processing is not efficient because:
    # - SAM model is cached and shared
    # - GPU operations are serialized anyway
    # - Multiple threads would just compete for GPU/CPU resources
    if method in ('sam_cpu', 'sam_gpu'):
        print(f"Note: SAM methods use a cached model, switching to sequential processing for efficiency.")
        return process_sequential(image_files, method=method, llm_censor=llm_censor, llm_max_iterations=llm_max_iterations, save_successful_params=False, llm_debug=llm_debug)
    
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
    
    # Note: LLM censor is not thread-safe, so we disable it for parallel processing
    # If LLM censor is enabled, we should use sequential processing
    if llm_censor and llm_censor.enabled:
        print("Note: LLM censor enabled. Switching to sequential processing for thread safety.")
        return process_sequential(image_files, method=method, llm_censor=llm_censor, llm_max_iterations=llm_max_iterations, save_successful_params=False, llm_debug=llm_debug)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(remove_background_single, img, method, None, llm_max_iterations, False, llm_debug): img 
            for img in image_files
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            image_file = future_to_file[future]
            
            try:
                result = future.result()
                if len(result) == 5:
                    success, input_path, output_path, error, debug_results = result
                else:
                    # Backward compatibility
                    success, input_path, output_path, error = result
                    debug_results = None
                
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
    parser.add_argument('--llm-censor', action='store_true',
                       help='Enable LLM censor for SAM methods (requires Ollama with vision model)')
    parser.add_argument('--llm-model', default='llava:13b',
                       help='Ollama model name for LLM censor (default: llava:13b)')
    parser.add_argument('--llm-url', default='http://localhost:11434',
                       help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--llm-iterations', type=int, default=3,
                       help='Maximum iterations for LLM parameter tuning (default: 3)')
    parser.add_argument('--save-successful-params', action='store_true',
                       help='Save successful SAM parameters for next time (only with LLM censor)')
    parser.add_argument('--llm-debug', action='store_true',
                       help='Debug mode: keep all intermediate results, show voting window')
    
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
    
    # Create LLM censor if requested (only for SAM methods)
    llm_censor = None
    if args.llm_censor:
        if args.method not in ('sam_cpu', 'sam_gpu'):
            print("Warning: LLM censor is only available for SAM methods (sam_cpu, sam_gpu).")
            print("LLM censor will be disabled.")
        elif LLM_CENSOR_AVAILABLE:
            try:
                llm_censor = create_llm_censor(
                    model_name=args.llm_model,
                    base_url=args.llm_url,
                    enabled=True
                )
                if llm_censor and llm_censor.enabled:
                    print(f"LLM censor enabled: {args.llm_model}")
                else:
                    print("Warning: LLM censor could not be initialized. Continuing without it.")
                    llm_censor = None
            except Exception as e:
                print(f"Warning: Failed to initialize LLM censor: {e}")
                print("Continuing without LLM censor.")
                llm_censor = None
        else:
            print("Warning: LLM censor module not available.")
            print("Install requests library: pip install requests")

    # Process images
    llm_debug = args.llm_debug if hasattr(args, 'llm_debug') else False
    if args.parallel:
        stats = process_parallel(valid_files, max_workers=args.workers, method=args.method,
                                llm_censor=llm_censor, llm_max_iterations=args.llm_iterations, save_successful_params=args.save_successful_params, llm_debug=llm_debug)
    else:
        stats = process_sequential(valid_files, method=args.method,
                                   llm_censor=llm_censor, llm_max_iterations=args.llm_iterations, save_successful_params=args.save_successful_params, llm_debug=llm_debug)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    if 'total_time' in stats:
        total_time = stats['total_time']
        avg_time = total_time / stats['total'] if stats['total'] > 0 else 0
        print(f"Total time: {total_time:.1f}s (avg: {avg_time:.1f}s per image)")
    print("=" * 60)
    
    # Only exit with error code if ALL images failed
    # If some succeeded, exit with 0 but print warning
    if stats['errors'] > 0:
        if stats['success'] == 0:
            # All failed - exit with error
            print(f"\n[ERROR] All {stats['total']} image(s) failed to process.")
            sys.exit(1)
        else:
            # Some succeeded, some failed - warn but don't exit with error
            print(f"\n[WARNING] {stats['errors']} of {stats['total']} image(s) failed, but {stats['success']} succeeded.")
            print("Check error messages above for details.")
            sys.exit(0)


if __name__ == "__main__":
    main()

