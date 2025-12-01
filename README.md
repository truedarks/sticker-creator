# Sticker Creator - Batch Image Processing GUI

A user-friendly Python application for batch processing images: removing backgrounds and creating stickers with white outlines. Features a simple drag-and-drop interface for Windows.

![GUI Preview](https://via.placeholder.com/800x400?text=Sticker+Creator+GUI)

## Features

- 🎨 **Smart Background Removal**: Automatically detects and removes backgrounds from images
  - **rembg CPU**: Fast CPU-based background removal (default)
  - **rembg GPU**: GPU-accelerated processing (requires CUDA)
  - **SAM**: Segment Anything Model for advanced segmentation
  - **LLM Censor** (NEW): AI-powered quality control for SAM using Ollama vision models
- ✨ **Sticker Creation**: Creates stickers with customizable white outlines
- 🚀 **Batch Processing**: Process multiple images simultaneously using parallel processing
- 🖱️ **Drag & Drop Interface**: Simple GUI with drag-and-drop support
- ⚡ **Fast Processing**: Uses all CPU cores for maximum speed
- 🎯 **Automatic Setup**: Installs required dependencies on first run
- 🔍 **Debug Mode**: Detailed logging for troubleshooting

## Quick Start

### Windows (Recommended)

1. **Download** the repository
2. **Double-click** `sticker_gui.bat` (or `sticker_gui.vbs`)
3. **Drag and drop** images onto the GUI:
   - **Left panel**: Remove background from images
   - **Right panel**: Create stickers with white outline

That's it! The application will:
- Check for required dependencies on first run
- Offer to install missing libraries automatically
- Process your images in the background

### Requirements

- **Python 3.8+** (must be in PATH)
- Windows OS (for .bat/.vbs launchers)

### First Run

On first launch, the application will check for required libraries:
- `rembg` - for background removal
- `tkinterdnd2` - for drag-and-drop support

If any are missing, you'll be prompted to install them. The installation is automatic and takes just a few minutes.

## Installation

### Automatic (Recommended)

Just run `sticker_gui.bat` - dependencies will be installed automatically if needed.

**Note:** All dependencies are installed locally in the `lib/site-packages` folder, making the project self-contained.

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/truedarks/sticker-creator.git
cd sticker-creator

# Install dependencies locally (recommended)
pip install --target lib/site-packages -r requirements.txt

# Or install globally (not recommended)
pip install -r requirements.txt

# Optional: Install drag-and-drop support
pip install --target lib/site-packages tkinterdnd2

# Run the GUI
python sticker_gui.py
```

### Local Dependencies

The project uses local dependencies stored in `lib/site-packages/`. This means:
- ✅ All dependencies are in the project folder
- ✅ No conflicts with other Python projects
- ✅ Easy to transfer to another computer
- ✅ Self-contained project

See `README_LOCAL.md` for more details about local dependency management.

## Usage

### GUI Mode (Recommended)

1. Launch `sticker_gui.bat`
2. **Remove Background**: Drag images to the left panel
3. **Create Stickers**: Drag images to the right panel
4. Click on panels to browse files if drag-and-drop is not available

### Command Line Mode

#### Background Removal

```bash
# Single image (CPU)
python batch_remove_bg.py image.png

# Multiple images (parallel, CPU)
python batch_remove_bg.py image1.png image2.png image3.png --parallel

# Using GPU acceleration
python batch_remove_bg.py *.png --parallel --method rembg_gpu

# Using SAM (Segment Anything Model) on CPU
python batch_remove_bg.py *.png --parallel --method sam_cpu

# Using SAM with GPU acceleration
python batch_remove_bg.py *.png --parallel --method sam_gpu

# Using SAM with LLM censor for quality control (requires Ollama)
python batch_remove_bg.py *.png --method sam_cpu --llm-censor --llm-model llava:13b

# Custom workers
python batch_remove_bg.py *.png --parallel --workers 4 --method rembg_cpu
```

#### Sticker Creation

```bash
# Single image (CPU)
python batch_create_sticker.py image.png

# Multiple images (parallel, CPU)
python batch_create_sticker.py image1.png image2.png image3.png --parallel

# Using GPU acceleration
python batch_create_sticker.py *.png --parallel --method rembg_gpu

# Using SAM (CPU)
python batch_create_sticker.py *.png --parallel --method sam_cpu

# Using SAM (GPU)
python batch_create_sticker.py *.png --parallel --method sam_gpu

# Custom outline width
python batch_create_sticker.py *.png --parallel --width 15 --method rembg_cpu
```

#### Single Image Processing

```bash
# Create sticker (CPU)
python create_sticker.py input.png output.png --width 20

# Create sticker with GPU
python create_sticker.py input.png output.png --width 20 --bg-method rembg_gpu

# Create sticker with SAM (CPU)
python create_sticker.py input.png output.png --width 20 --bg-method sam_cpu

# Create sticker with SAM (GPU)
python create_sticker.py input.png output.png --width 20 --bg-method sam_gpu

# Remove background only (CPU)
python remove_bg.py input.png output.png

# Remove background with GPU
python remove_bg.py input.png output.png --method rembg_gpu

# Remove background with SAM (CPU)
python remove_bg.py input.png output.png --method sam_cpu

# Remove background with SAM (GPU)
python remove_bg.py input.png output.png --method sam_gpu
```

## Output Files

- **Background Removal**: Creates files with `_nobg` suffix (e.g., `image_nobg.png`)
- **Sticker Creation**: Creates files with `_sticker` suffix (e.g., `image_sticker.png`)

## Supported Formats

- PNG (recommended)
- JPG/JPEG
- BMP
- TIFF
- WebP

## How It Works

### Background Removal
1. Analyzes image for background presence
2. Uses AI-powered `rembg` library to remove background
3. Outputs image with transparent background

### Sticker Creation
1. Removes background (if present)
2. Adds white outline around the subject
3. Applies edge smoothing for polished look
4. Crops to optimal size

### LLM Censor (SAM Quality Control)
When using SAM methods with `--llm-censor` flag:
1. **Parallel Analysis**: LLM analyzes the original image to identify the main object
2. **Segmentation**: SAM generates masks to remove background
3. **Quality Evaluation**: LLM evaluates the segmentation result
4. **Iterative Improvement**: If quality is insufficient, LLM suggests parameter adjustments and SAM retries
5. **Final Output**: Best result is saved after up to 3 iterations

## Performance

- **Parallel Processing**: Uses all CPU cores (except one) for maximum speed
- **Sequential Processing**: Processes one image at a time (safer for low-end systems)
- **Auto-detection**: Automatically detects CPU core count

Example speeds:
- 16-core CPU → uses 15 workers
- 8-core CPU → uses 7 workers
- 4-core CPU → uses 3 workers

## Dependencies

Core dependencies (auto-installed):
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `scipy` - Scientific computing
- `rembg` - Background removal AI
- `onnxruntime` - Required by rembg

Optional:
- `tkinterdnd2` - Drag-and-drop support (recommended)
- `requests` - Required for LLM censor (Ollama API)
- `segment-anything` - Required for SAM methods
- `torch`, `torchvision` - Required for SAM methods
- `opencv-python` - Required for SAM methods

All dependencies are listed in `requirements.txt`.

## LLM Censor Setup

The LLM censor feature uses Ollama with vision models to improve SAM segmentation quality.

### Prerequisites
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Start Ollama**: Run `ollama serve` (usually runs automatically)
3. **Install Vision Model**: 
   ```bash
   ollama pull llava:13b
   # or
   ollama pull llava-next:latest
   ```

### Recommended Models
- **llava:13b** (recommended): Best balance of quality and speed
- **llava-next:latest**: Latest version with improved vision understanding
- **llava:7b**: Faster but less accurate
- **qwen2.5-vl**: Alternative vision model

### Usage Examples

```bash
# Basic usage with LLM censor
python batch_remove_bg.py image.png --method sam_cpu --llm-censor

# Custom model and iterations
python batch_remove_bg.py image.png --method sam_cpu --llm-censor --llm-model llava-next:latest --llm-iterations 5

# Custom Ollama URL (if running on different host/port)
python batch_remove_bg.py image.png --method sam_cpu --llm-censor --llm-url http://localhost:11434
```

### How LLM Censor Works
1. **Image Analysis**: LLM analyzes the original image to understand the main subject
2. **Segmentation**: SAM removes background using current parameters
3. **Quality Check**: LLM evaluates if the segmentation captured the object correctly
4. **Parameter Tuning**: If quality is low, LLM suggests adjustments to SAM parameters:
   - `points_per_side`: More points = more detailed masks
   - `pred_iou_thresh`: Higher = stricter mask selection
   - `stability_score_thresh`: Higher = more stable masks
   - `min_mask_region_area`: Higher = filters small artifacts
5. **Retry**: SAM regenerates masks with new parameters
6. **Best Result**: After up to 3 iterations, the best result is saved

### Notes
- LLM censor only works with SAM methods (`sam_cpu`, `sam_gpu`)
- Parallel processing is automatically disabled when LLM censor is enabled (for thread safety)
- Processing time increases with LLM censor (each iteration requires LLM evaluation)
- Make sure Ollama is running before using LLM censor

## Troubleshooting

### Python not found
- Make sure Python 3.8+ is installed
- Add Python to your system PATH
- Restart your computer after installation

### Drag-and-drop not working
- Install `tkinterdnd2`: `pip install tkinterdnd2`
- Or use the "Browse Files" button instead

### Installation fails
- Run as administrator
- Check your internet connection
- Install manually: `pip install rembg onnxruntime tkinterdnd2`

### Processing errors
- Check that images are valid image files
- Ensure sufficient disk space
- Try processing one image at a time

### Debug mode
To enable detailed logging for troubleshooting:
```bash
# Windows PowerShell
$env:STICKER_DEBUG="1"
python sticker_gui.py

# Windows CMD
set STICKER_DEBUG=1
python sticker_gui.py

# Linux/Mac
export STICKER_DEBUG=1
python sticker_gui.py
```

Debug logs will show:
- Library import status
- GPU availability checks
- Detailed error messages with stack traces
- Processing steps for each method

## LLM Censor Setup

The LLM censor feature uses Ollama with vision models to improve SAM segmentation quality.

### Prerequisites
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Start Ollama**: Run `ollama serve` (usually runs automatically)
3. **Install Vision Model**: 
   ```bash
   ollama pull llava:13b
   # or
   ollama pull llava-next:latest
   ```

### Recommended Models
- **llava:13b** (recommended): Best balance of quality and speed
- **llava-next:latest**: Latest version with improved vision understanding
- **llava:7b**: Faster but less accurate
- **qwen2.5-vl**: Alternative vision model

### Usage Examples

```bash
# Basic usage with LLM censor
python batch_remove_bg.py image.png --method sam_cpu --llm-censor

# Custom model and iterations
python batch_remove_bg.py image.png --method sam_cpu --llm-censor --llm-model llava-next:latest --llm-iterations 5

# Custom Ollama URL (if running on different host/port)
python batch_remove_bg.py image.png --method sam_cpu --llm-censor --llm-url http://localhost:11434
```

### How LLM Censor Works
1. **Image Analysis**: LLM analyzes the original image to understand the main subject
2. **Segmentation**: SAM removes background using current parameters
3. **Quality Check**: LLM evaluates if the segmentation captured the object correctly
4. **Parameter Tuning**: If quality is low, LLM suggests adjustments to SAM parameters:
   - `points_per_side`: More points = more detailed masks
   - `pred_iou_thresh`: Higher = stricter mask selection
   - `stability_score_thresh`: Higher = more stable masks
   - `min_mask_region_area`: Higher = filters small artifacts
5. **Retry**: SAM regenerates masks with new parameters
6. **Best Result**: After up to 3 iterations, the best result is saved

### Notes
- LLM censor only works with SAM methods (`sam_cpu`, `sam_gpu`)
- Parallel processing is automatically disabled when LLM censor is enabled (for thread safety)
- Processing time increases with LLM censor (each iteration requires LLM evaluation)
- Make sure Ollama is running before using LLM censor

## Project Structure

```
sticker-creator/
├── sticker_gui.py          # Main GUI application
├── sticker_gui.bat          # Windows launcher (no console)
├── sticker_gui.vbs         # VBScript launcher
├── bg_removal.py          # Unified background removal module
├── llm_censor.py          # LLM censor for SAM quality control
├── batch_remove_bg.py      # Batch background removal
├── batch_create_sticker.py # Batch sticker creation
├── create_sticker.py       # Single sticker creation
├── remove_bg.py            # Single background removal
└── requirements.txt        # Python dependencies
```

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## Credits

- Uses [rembg](https://github.com/danielgatis/rembg) for background removal
- Built with Python and tkinter
