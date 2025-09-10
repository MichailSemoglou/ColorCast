# ColorCast

A PyQt5 GUI application for style transfer using histogram matching (LUT transfer) between images. This tool allows you to transfer the color characteristics and mood from one image (style) to another image (content) automatically.

## Interface

![ColorCast Interface](interface.png)

_ColorCast's intuitive interface showing content image, style image, and result panels_

## Features

- **Easy-to-use GUI** with intuitive button-based interface
- **Automatic image resizing** - works with images of different dimensions
- **Multiple format support** - PNG, JPG, JPEG, BMP, TIF, TIFF
- **Smart image preprocessing** - automatically handles grayscale and RGBA images
- **Real-time preview** of content, style, and result images
- **Robust error handling** for corrupted or unsupported files
- **High-quality output** with preserve aspect ratio scaling

## How It Works

The application uses **histogram matching** to transfer color characteristics:

1. **Load Content Image**: The image you want to transform
2. **Load Style Image**: The image whose color palette/mood you want to copy
3. **Apply Transfer**: The algorithm matches color histograms between the images
4. **Save Result**: Export your color-graded image

The algorithm analyzes the color distribution (histogram) of both images and creates a mapping that transforms the content image's colors to match the style image's color distribution, while preserving the original structure and details.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/MichailSemoglou/ColorCast.git
cd ColorCast
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python colorcast.py
```

### Step-by-step:

1. Click "Load Content Image" to select your base image
2. Click "Load Style Image" to select the image whose style you want to copy
3. Click "Apply Style Transfer" to process the images
4. Click "Save Result" to export the final image
5. Use "Clear Images" to reset and start over

## Requirements

- Python 3.7+
- NumPy
- scikit-image
- scipy
- PyQt5

See `requirements.txt` for specific versions.

## Examples

The application works great for:

- **Film color grading** (matching scenes shot at different times)
- **Photography** (applying vintage or cinematic looks)
- **Art creation** (transferring painting styles to photos)
- **Social media content** (consistent color themes)

## Technical Details

- Uses `scikit-image`'s histogram matching algorithm
- Processes RGB channels independently
- Automatically converts grayscale images to RGB (duplicates channel)
- Strips alpha channels from RGBA images for processing
- Automatically handles EXIF orientation data
- Preserves image quality with anti-aliasing during resize operations

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).
