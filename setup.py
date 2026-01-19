from setuptools import setup, find_packages

setup(
    name="transformer-viz",
    version="0.1.0",
    description="Visualization library for transformer architectures, compatible with TransformerLens",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "transformerlens": ["transformer-lens>=1.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
