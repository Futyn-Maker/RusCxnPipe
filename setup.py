"""Setup script for RusCxnPipe library."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip()
                    and not line.startswith('#')]

setup(
    name="ruscxnpipe",
    version="0.1.0",
    author="Andrey Yakuboy",
    author_email="github@yakuboy.ru",  # Replace with your email
    description="Russian Constructicon Pattern Extraction Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Futyn-Maker/ruscxnpipe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Russian",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "demo": [
            "gradio>=3.0",
            "streamlit>=1.0",
        ],
    },
    package_data={
        "ruscxnpipe": [
            "data/*.json",
            "data/*.txt",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "nlp",
        "russian",
        "constructicon",
        "linguistics",
        "pattern-extraction",
        "text-analysis",
        "construction-grammar",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Futyn-Maker/ruscxnpipe/issues",
        "Source": "https://github.com/Futyn-Maker/ruscxnpipe",
        "Documentation": "https://github.com/Futyn-Maker/ruscxnpipe#readme",
    },
)
