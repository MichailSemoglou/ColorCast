# ColorCast

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-44%25-orange.svg)](tests/)
[![Tests](https://img.shields.io/badge/tests-185%20passed-brightgreen.svg)](tests/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18550038.svg)](https://doi.org/10.5281/zenodo.18550038)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/colorcast?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/colorcast)

**ColorCast** is a Python toolkit for color and style transfer between images. It provides histogram matching, mean/std transfer, LUT-based curves (linear, S-curve, contrast), and selective regional color transfer across shadows, midtones, and highlights. Color-vision-deficiency simulation covers deuteranopia, protanopia, and tritanopia; the Daltonization pipeline re-encodes the chromatic information these deficiencies would otherwise hide, shifting it into a channel the affected observer can still perceive.

## Features

### Core Capabilities

- **9 Transfer Methods** - Choose from multiple algorithms for different effects
- **Intensity Control** - Smooth slider (0-100%) to blend between original and styled images
- **Selective Regional Transfer** - Target shadows, midtones, or highlights specifically
- **Modular Package** - Use as Python API or run GUI/CLI
- **Performance Optimized** - LRU caching and batch processing support
- **Tested** - 44% test coverage with 185 passing tests
- **Documented** - API documentation and examples
- **Plugin Architecture** - Easy to add custom transfer methods

### Transfer Methods

1. **Histogram Matching** - Classic histogram equalization, preserves local contrast
2. **Mean/Std Transfer** - Statistical color matching using mean and standard deviation
3. **Lab Color Transfer (Reinhard)** - Industry-standard perceptually uniform color transfer in L*a*b\* space
4. **LUT + Linear Curve** - Histogram matching with linear tone mapping
5. **LUT + S-Curve** - Adds smooth contrast enhancement to histogram matching
6. **LUT + Contrast** - Increases overall contrast with power curve
7. **Selective: Shadows** - Transfers colors only in dark regions (luminance < 0.3)
8. **Selective: Midtones** - Transfers colors only in mid-tones (0.3 to 0.7)
9. **Selective: Highlights** - Transfers colors only in bright regions (luminance > 0.7)

## Interface

![ColorCast Interface](imgs/interface.png)

_The ColorCast interface: content image, style image, and result panels_

![ColorCast Daltonize (P) mode](imgs/interface_2.png)

_Daltonize (P) mode correcting an image for protanopia, with correction intensity control_

## Installation

### From PyPI (Recommended)

Install ColorCast using pip:

```bash
pip install colorcast
```

For GPU support (requires CUDA):

```bash
pip install colorcast[gpu]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/MichailSemoglou/ColorCast.git
cd ColorCast

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# For development with all dependencies
pip install -e ".[dev]"

# For GPU support (requires CUDA)
pip install -e ".[gpu]"
```

### Requirements

- Python 3.10+
- NumPy >= 1.20.0
- scikit-image >= 0.19.0
- PyQt5 >= 5.15.0 (for GUI)
- scipy >= 1.7.0

See `pyproject.toml` for complete dependency specifications.

## Usage

### 1. GUI Application

Run the graphical interface:

```bash
colorcast-gui
```

Or use the package entry point:

```bash
python -m colorcast
```

The root `colorcast.py` script is kept as a deprecated compatibility shim and also launches the GUI. Use `python -m colorcast` or `colorcast-gui` for new work.

**Step-by-step:**

1. **Select Transfer Method**: Choose from 15 methods at the top: 9 transfer algorithms, 3 color-vision-deficiency simulators, and 3 Daltonization corrections
2. **Load Content Image**: Click to select your base image
3. **Load Style Image**: Click to select the image whose style you want to copy (not required for simulator and correction modes)
4. **Apply Style Transfer**: Click to process the images
5. **Adjust Intensity**: Use the slider to control effect strength (0-100%)
6. **Save Result**: Export your final image in your preferred format
7. **Clear Images**: Reset and start over with new images

### 2. Command-Line Interface

Basic color transfer:

```bash
colorcast transfer content.jpg style.jpg -o output.jpg
```

With method and intensity:

```bash
colorcast transfer content.jpg style.jpg -o output.jpg -m meanstd -i 0.7
```

Simulate a color-vision deficiency (no style image needed):

```bash
colorcast transfer content.jpg -m simulate_protanopia -o output.png
```

Batch process directory:

```bash
colorcast batch ./content_dir style.jpg -o ./output_dir
```

List available methods:

```bash
colorcast list-methods
```

Get package information:

```bash
colorcast info --version
```

**CLI Options:**

- `-m, --method`: Transfer method (histogram, meanstd, lab_reinhard, lut_linear, lut_scurve, lut_contrast, selective_shadows, selective_midtones, selective_highlights, simulate_protanopia, simulate_deuteranopia, simulate_tritanopia)
- `-i, --intensity`: Blend intensity 0.0-1.0
- `-w, --workers`: Number of parallel workers (default: 4)
- `-p, --pattern`: File pattern to match (default: \*.jpg)

### 3. Python API

```python
from colorcast import load_image, match_histograms_multichannel, save_image

# Load images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Apply histogram matching
result = match_histograms_multichannel(content, style)

# Save result
save_image(result, "output.jpg")
```

#### Using Different Methods

```python
from colorcast import (
    load_image,
    color_transfer_meanstd,
    color_transfer_lab,
    lut_transfer_with_curve,
    selective_color_transfer,
    blend_images,
    save_image,
)

content = load_image("content.jpg")
style = load_image("style.jpg")

# Mean/Std transfer
result1 = color_transfer_meanstd(content, style)

# Lab color transfer (Reinhard method)
result_lab = color_transfer_lab(content, style, alpha=0.8)

# LUT with S-curve
result2 = lut_transfer_with_curve(content, style, "s-curve")

# Selective shadows transfer
result3 = selective_color_transfer(content, style, mode="shadows")

# Blend with 70% intensity
final = blend_images(content, result2, intensity=0.7)
save_image(final, "output.jpg")
```

#### Using the Plugin Registry

```python
from colorcast import load_image, registry

# List available methods
methods = registry.list_methods()
print(f"Available methods: {methods}")

# Get method and use it
content = load_image("content.jpg")
style = load_image("style.jpg")

method = registry.get_method("histogram")
result = method.transfer(content, style)
```

#### Batch Processing

```python
from colorcast.processing.batch import BatchProcessor
from colorcast import match_histograms_multichannel

# Create batch processor
processor = BatchProcessor(
    transfer_method=match_histograms_multichannel,
    max_workers=4,
)

# Process directory
results = processor.process_directory(
    content_dir="./content_images",
    style_image="style.jpg",
    output_dir="./output",
    pattern="*.jpg",
)

# Check for failed files
if processor.failed_files:
    print(f"Failed to process {len(processor.failed_files)} files")
```

#### Using Caching

```python
from colorcast.processing.cache import LRUCache

# Create cache
cache = LRUCache(max_size=100)

# Use cache for expensive operations
result = cache.get_or_compute(
    key="transfer_key",
    compute_func=lambda: match_histograms_multichannel(content, style),
)

# Get cache statistics
stats = cache.stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

## Tips for Best Results

### Choosing the Right Method

- **Histogram Matching**: Best for artistic effects and dramatic color shifts
- **Mean/Std Transfer**: Better for subtle, natural-looking color grading
- **Lab Color Transfer (Reinhard)**: Industry-standard method for professional color grading with perceptually uniform results
- **LUT Curves**: Experiment with different curves for varied contrast effects
  - Linear: Standard transfer
  - S-Curve: Enhanced midtones
  - Contrast: Punchier overall look
- **Selective Transfer**: Target specific tonal ranges for precise control
  - Shadows: Affect only dark areas
  - Midtones: Affect only mid-tones (most natural for skin tones)
  - Highlights: Affect only bright areas

### Intensity Blending

- **0-30%**: Very subtle color correction
- **30-60%**: Natural color grading
- **60-80%**: Noticeable style transfer
- **80-100%**: Full style application

### Image Preparation

- Use images with similar aspect ratios for best results
- Higher resolution images produce better quality transfers
- For selective transfer, ensure good dynamic range in both images

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=colorcast --cov-report=html

# Run specific test file
pytest tests/test_transfer_methods.py

# Run with verbose output
pytest -v
```

**Test Coverage:** 44% (185 passing tests, 1 skipped)

- Core transfer methods: 99% coverage
- Full test suite including integration, performance, and property-based tests
- Property-based tests require the optional `hypothesis` dependency

## Technical Details

### Algorithms Implemented

**1. Histogram Matching**

```python
matched = exposure.match_histograms(source, reference)
```

- Per-channel histogram equalization
- Preserves complete color distribution
- Best for dramatic color transformations

**2. Mean/Standard Deviation Transfer**

```python
result = ((source - μ_source) × (σ_ref / σ_source)) + μ_ref
```

- Matches statistical properties per channel
- Better color balance for photographic work

**3. Lab Color Space Transfer (Reinhard)**

```python
result_lab = ((source_lab - μ_source) × (σ_ref / σ_source)) + μ_ref
```

- Operates in perceptually uniform L*a*b\* color space
- Industry-standard in professional color grading
- Preserves color relationships better than RGB methods
- More natural-looking results

**4. LUT with Curves**

- **Linear**: Standard histogram matching
- **S-Curve**: `0.5 + 0.5 × sin(π(x - 0.5))` - smooth midtone enhancement
- **Contrast**: `x^0.8` - power curve for increased punch

**5. Selective Color Transfer**

```python
luminance = 0.299R + 0.587G + 0.114B
mask = (luminance >= shadow_threshold) & (luminance <= highlight_threshold)
result = source × (1 - mask) + matched × mask
```

- Region-based masking using luminance
- Precise tonal range targeting

### Performance Features

- **LRU Cache**: 10-20x speedup for repeated operations
- **Batch Processing**: Parallel processing with ThreadPoolExecutor
- **Memory Efficient**: Processes images in-place where possible
- **Debounced Updates**: 50ms delay prevents UI blocking
- **Automatic Format Conversion**: Grayscale → RGB, RGBA → RGB (alpha composited onto a white background)

## Use Cases

ColorCast is perfect for:

- **Film Color Grading** - Matching scenes shot at different times
- **Photography** - Applying vintage or cinematic looks
- **Art Creation** - Transferring painting styles to photos
- **Social Media Content** - Creating consistent color themes
- **Game Development** - Style transfer for game assets
- **Research** - Academic research in color transfer

## Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For development:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black colorcast/
isort colorcast/
mypy colorcast/

# Run type checking
mypy colorcast/
```

## License

Released under the [MIT License](LICENSE).

## Author

**Michail Semoglou**

- Email: m.semoglou@tongji.edu.cn

## Acknowledgments

- Built with [scikit-image](https://scikit-image.org/) for image processing
- GUI powered by [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- Performance optimized with [NumPy](https://numpy.org/)

## Support

For issues, questions, or suggestions:

- [Report a bug](https://github.com/MichailSemoglou/ColorCast/issues)
- [Request a feature](https://github.com/MichailSemoglou/ColorCast/issues)
