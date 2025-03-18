from setuptools import setup, find_packages

setup(
    name="time_forecaster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "yfinance>=0.1.63",
        "tqdm>=4.62.0"
    ],
    python_requires=">=3.7",
) 