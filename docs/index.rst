ColorCast Documentation
======================

.. image:: https://img.shields.io/badge/version-2.6.0-blue.svg
   :target: https://github.com/MichailSemoglou/ColorCast
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.10%2B-green.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

ColorCast is a Python library for color and style transfer between images,
providing multiple algorithms, CVD simulation, Daltonization, and analysis tools.

**Features:**

* Multiple color transfer algorithms (histogram matching, mean/std, Lab space, LUT-based)
* Color-vision-deficiency simulation (protanopia, deuteranopia, tritanopia)
* Daltonization correction to re-encode lost chromatic information
* Selective color transfer (shadows, midtones, highlights)
* Batch processing and LRU caching
* Analysis tools (PSNR, SSIM, color metrics, error maps)
* CLI, GUI, and Python library API

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


Installation
------------

**Core Installation:**

.. code-block:: bash

    pip install colorcast

**From source:**

.. code-block:: bash

    git clone https://github.com/MichailSemoglou/ColorCast.git
    cd ColorCast
    pip install -e ".[dev,analysis]"

License
=======

This project is licensed under the MIT License — see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`