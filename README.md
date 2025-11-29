# Sticker Creator - Batch Image Processing GUI

A user-friendly Python application for batch processing images: removing backgrounds and creating stickers with white outlines. Features a simple drag-and-drop interface for Windows.

![GUI Preview](https://via.placeholder.com/800x400?text=Sticker+Creator+GUI)

## Features

- 🎨 **Smart Background Removal**: Automatically detects and removes backgrounds from images
- ✨ **Sticker Creation**: Creates stickers with customizable white outlines
- 🚀 **Batch Processing**: Process multiple images simultaneously using parallel processing
- 🖱️ **Drag & Drop Interface**: Simple GUI with drag-and-drop support
- ⚡ **Fast Processing**: Uses all CPU cores for maximum speed
- 🎯 **Automatic Setup**: Installs required dependencies on first run

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

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/truedarks/sticker-creator.git
cd sticker-creator

# Install dependencies
pip install -r requirements.txt

# Optional: Install drag-and-drop support
pip install tkinterdnd2

# Run the GUI
python sticker_gui.py
```

## Usage

### GUI Mode (Recommended)

1. Launch `sticker_gui.bat`
2. **Remove Background**: Drag images to the left panel
3. **Create Stickers**: Drag images to the right panel
4. Click on panels to browse files if drag-and-drop is not available

### Command Line Mode

#### Background Removal

```bash
# Single image
python batch_remove_bg.py image.png

# Multiple images (parallel)
python batch_remove_bg.py image1.png image2.png image3.png --parallel

# Custom workers
python batch_remove_bg.py *.png --parallel --workers 4
```

#### Sticker Creation

```bash
# Single image
python batch_create_sticker.py image.png

# Multiple images (parallel)
python batch_create_sticker.py image1.png image2.png image3.png --parallel

# Custom outline width
python batch_create_sticker.py *.png --parallel --width 15
```

#### Single Image Processing

```bash
# Create sticker
python create_sticker.py input.png output.png --width 20

# Remove background only
python remove_bg.py input.png output.png
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

All dependencies are listed in `requirements.txt`.

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

## Project Structure

```
sticker-creator/
├── sticker_gui.py          # Main GUI application
├── sticker_gui.bat          # Windows launcher (no console)
├── sticker_gui.vbs         # VBScript launcher
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
