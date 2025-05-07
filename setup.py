from setuptools import setup, find_packages

setup(
    name="qaravan",
    packages=find_packages(where="src"),
    python_requires='>=3.11',
    package_dir={"": "src"},
    version="0.1.36", 
    author="Faisal Alam",
    author_email="mfalam2@illinois.edu",
    description="Unified classical simulation of noiseless and noisy quantum circuits",
    install_requires=[
    "numpy>=1.20,<2.0",
    "scipy>=1.15.2",
    "matplotlib>=3.10.1",
    "scienceplots>=2.0",
    "tqdm>=4.67.1",
    "ncon_torch>=0.3",
    "torch>=1.12",
    ]
)
