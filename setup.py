from setuptools import setup, find_packages

setup(
    name="qaravan",
    packages=find_packages(where="src"),
    python_requires='>=3.8',
    package_dir={"": "src"},
    version="0.1.46", 
    author="Faisal Alam",
    author_email="mfalam2@illinois.edu",
    description="Unified classical simulation of noiseless and noisy quantum circuits",
    install_requires=[
    "numpy>=1.20,<2.0",
    "scipy>=1.5.2,<2.0",
    "matplotlib>=3.7,<4.0",
    "tqdm>=4.67.1",
    "ncon_torch>=0.3",
    "torch>=1.12",
    "stim>=1.14.0"
]
)
