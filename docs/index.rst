ColorCast Documentation
=====================

.. image:: https://img.shields.io/badge/version-2.4.2-blue.svg
   :target: https://github.com/MichailSemoglou/ColorCast
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.10%2B-green.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

ColorCast is a Python library for color and style transfer between images,
providing multiple algorithms, GPU acceleration, and analysis tools.

**Features:**

* Multiple color transfer algorithms (histogram matching, mean/std, Lab space, LUT-based)
* GPU acceleration with CuPy (10-50x speedup on supported hardware)
* Selective color transfer (shadows, midtones, highlights)
* Blending and intensity control
* Batch processing for multiple images
* Caching for performance optimization
* Analysis tools (PSNR, SSIM, color metrics)
* Visualization utilities for comparing methods
* Property-based testing for correctness

**Quick Start**

.. code-block:: python

    from colorcast import load_image, color_transfer_meanstd, save_image
    
    # Load images
    content = load_image("content.jpg")
    style = load_image("style.jpg")
    
    # Apply color transfer
    result = color_transfer_meanstd(content, style)
    
    # Save result
    save_image(result, "output.jpg")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   user_guide
   api/index
   advanced/index
   contributing


Installation
------------

**Core Installation:**

.. code-block:: bash

    pip install colorcast

**Full Installation (with development tools):**

.. code-block:: bash

    git clone https://github.com/MichailSemoglou/ColorCast.git
    cd ColorCast
    pip install -e .
    pip install -r requirements-dev.txt

**GPU Support (optional):**

.. code-block:: bash

    pip install cupy-cuda12x  # For CUDA 12.x
    # or
    pip install cupy-cuda11x  # For CUDA 11.x


Documentation Structure
---------------------

* :doc:`quickstart` - Get started with basic usage
* :doc:`user_guide` - Comprehensive user guide
* :doc:`api/index` - API reference
* :doc:`advanced/index` - Advanced topics and academic tools

Academic Use
============

ColorCast is designed for both practical applications and academic research.
The library includes:

* Quantitative metrics (PSNR, SSIM, color distance)
* Method comparison tools
* Visualization utilities
* Property-based testing for correctness verification

See :doc:`advanced/comparison` for details.

Performance
===========

* **CPU Processing:** Optimized NumPy operations
* **GPU Acceleration:** CuPy support for 10-50x speedup
* **Batch Processing:** Parallel processing for multiple images
* **Caching:** LRU cache for repeated operations

See :doc:`advanced/performance` for benchmarks.

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`