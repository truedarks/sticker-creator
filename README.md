# Sticker Creator with White Outline

A Python tool for creating stickers with white outlines from images. Automatically removes background if present and adds a customizable white border.

## Features

- ✅ **Smart Background Removal**: Automatically detects and removes background from images
- ✅ **White Outline**: Creates a white border that follows the image shape
- ✅ **Smooth Edges**: Optional smoothing for more polished outlines
- ✅ **Easy to Use**: Drag and drop interface via .bat files (Windows)
- ✅ **Transparent Background**: Output images have transparent backgrounds
- ✅ **Optional Ollama Integration**: Use Ollama LLM for image analysis

## Quick Start

### Windows (Easiest)

1. **Create Sticker**: Drag and drop a PNG file onto `create_sticker.bat`
2. **Remove Background Only**: Drag and drop a PNG file onto `remove_bg.bat`
3. **Batch Process (Sequential)**: Drag and drop multiple images onto `batch_remove_bg_sequential.bat`
4. **Batch Process (Parallel)**: Drag and drop multiple images onto `batch_remove_bg_parallel.bat`

That's it! The script will automatically:
- Detect if background removal is needed
- Install required libraries on first run (with your permission)
- Create the sticker with white outline

### Manual Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python create_sticker.py input.png output.png
   ```

## Usage

### Command Line

```bash
# Basic usage (20px outline, with smoothing)
python create_sticker.py input.png output.png

# Custom outline width
python create_sticker.py input.png output.png --width 30

# Without smoothing
python create_sticker.py input.png output.png --no-smooth

# Disable automatic background removal
python create_sticker.py input.png output.png --no-remove-bg

# Use Ollama for analysis
python create_sticker.py input.png output.png --use-ollama
```

### Remove Background Only

```bash
# Remove background only (no outline)
python remove_bg.py input.png

# Specify output file
python remove_bg.py input.png output.png
```

### As Python Module

```python
from create_sticker import create_sticker

create_sticker('input.png', 'output.png', outline_width=20, smooth=True)
```

## Parameters

- `--width` / `-w`: White outline width in pixels (default: 20)
- `--no-smooth`: Disable outline smoothing
- `--no-remove-bg`: Disable automatic background removal
- `--use-ollama`: Use Ollama for image analysis (requires running Ollama)

## Requirements

- Python 3.8+
- Pillow
- NumPy
- SciPy
- rembg (for background removal)
- onnxruntime (required by rembg)
- requests (for optional Ollama integration)

All dependencies are listed in `requirements.txt`.

## First Run

On first run, if `rembg` is not installed, you'll be prompted to install it. You can choose:
1. **System-wide installation** (default) - installs for all users
2. **User-only installation** - installs only for current user
3. **Cancel** - skip background removal feature

The library will be downloaded and installed automatically.

## Ollama Integration (Optional)

To use Ollama for image analysis:

1. Install [Ollama](https://ollama.ai/)
2. Pull a vision model:
   ```bash
   ollama pull llava:7b
   ```
3. Use the `--use-ollama` flag when running the script

**Recommended Ollama models:**
- `llava:7b` - Fast, good quality (recommended)
- `llava:13b` - More accurate, slower
- `llava:34b` - Most accurate, requires more memory

**Note:** Ollama is optional. Background removal works without it using `rembg`.

## How It Works

1. **Background Detection**: Checks if image has a background
2. **Background Removal** (if needed): Uses `rembg` to remove background, leaving only the subject on transparent background
3. **Outline Creation**: Adds white outline around the image
4. **Smoothing** (optional): Applies smoothing for smoother edges
5. **Output**: Saves final sticker with transparent background

## Batch Processing

For processing multiple images at once:

### Sequential Processing
- Drag and drop multiple images onto `batch_remove_bg_sequential.bat`
- Images are processed one after another
- Safer for systems with limited resources
- Slower but more stable

### Parallel Processing
- Drag and drop multiple images onto `batch_remove_bg_parallel.bat`
- Images are processed simultaneously using multiple CPU cores
- Uses all CPU cores except one (leaves one free to prevent system freeze)
- Much faster for large batches
- Automatically detects CPU core count

**Example:**
- 16-core CPU → uses 15 workers
- 8-core CPU → uses 7 workers
- 4-core CPU → uses 3 workers

### Command Line

```bash
# Sequential processing
python batch_remove_bg.py image1.png image2.png image3.png

# Parallel processing
python batch_remove_bg.py image1.png image2.png image3.png --parallel

# Custom number of workers
python batch_remove_bg.py *.png --parallel --workers 4
```

## Files

- `create_sticker.py` - Main script for creating stickers
- `remove_bg.py` - Script for background removal only
- `batch_remove_bg.py` - Batch processing script
- `create_sticker.bat` - Windows batch file (drag & drop)
- `remove_bg.bat` - Windows batch file for background removal
- `batch_remove_bg_sequential.bat` - Batch processing (sequential)
- `batch_remove_bg_parallel.bat` - Batch processing (parallel)
- `create_sticker_gui.bat` - GUI version with file picker
- `requirements.txt` - Python dependencies

## Examples

```bash
# Create sticker from image with background
python create_sticker.py character.jpg character_sticker.png

# Create sticker with 15px outline
python create_sticker.py moose.png moose_sticker.png -w 15

# Remove background only
python remove_bg.py photo.jpg photo_nobg.png
```

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

