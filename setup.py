"""
Setup script for Financial Sentiment Analysis AI
Professional installation and configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="financial-sentiment-ai",
    version="1.0.0",
    author="Financial Sentiment AI Team",
    author_email="team@financial-sentiment-ai.com",
    description="Professional financial sentiment analysis with VADER + FinBERT",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-sentiment-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "api": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
        "mongodb": [
            "pymongo>=4.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-sentiment=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "financial_sentiment_ai": [
            "config/*.yaml",
            "data/.gitkeep",
        ],
    },
    keywords=[
        "finance",
        "sentiment analysis", 
        "nlp",
        "fintech",
        "news analysis",
        "machine learning",
        "bert",
        "vader",
        "financial markets"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/financial-sentiment-ai/issues",
        "Source": "https://github.com/yourusername/financial-sentiment-ai",
        "Documentation": "https://financial-sentiment-ai.readthedocs.io/",
    },
)
