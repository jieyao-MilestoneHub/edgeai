"""
Setup script for EdgeAI Tuning SDK
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edgeai-tuning-sdk",
    version="1.0.0",
    author="EdgeAI Lab",
    author_email="lab@edgeai.example.com",
    description="Python SDK for EdgeAI Tuning Service API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edgeai/tuning-sdk",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "requests>=2.31.0",
        "urllib3>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # 可以添加命令行工具
            # "tuning-cli=tuning_sdk.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/edgeai/tuning-sdk/issues",
        "Source": "https://github.com/edgeai/tuning-sdk",
        "Documentation": "https://docs.edgeai.example.com",
    },
)
