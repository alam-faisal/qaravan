from setuptools import setup, find_packages

setup(
    name="qaravan",
    packages=find_packages(where="src"),
    python_requires='>=3.11',
    package_dir={"": "src"},
    version="0.1.17", 
    author="Faisal Alam",
    author_email="mfalam2@illinois.edu",
    description="Unified classical simulation of noiseless and noisy quantum circuits",
    install_requires=[
    "numpy>=2.2.4",
    "scipy>=1.15.2",
    "matplotlib>=3.10.1",
    "tqdm>=4.67.1",
    "ncon>=2.0.0"
    ]
)
