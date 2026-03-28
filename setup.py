from setuptools import setup, find_packages

# Setup configuration for the Gauge OCR package
# This allows installing the package in development mode with 'pip install -e .'
setup(
    name="gauge-ocr",
    version="0.1.0",
    # Automatically discover all packages in the directory
    packages=find_packages(),
    # List of required dependencies
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "opencv-python",
        "numpy",
        "scikit-image",
        "timm",
        "pyyaml",
        "tqdm",
        "matplotlib",
    ],
    author="Gauge OCR Team",
    description="Spatial Geometry-Aware Vision Large Model for Gauge OCR",
    keywords="ocr, gauge, computer-vision, transformers, deep-learning",
)
