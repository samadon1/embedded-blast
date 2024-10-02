from setuptools import setup, find_packages
import os

# Find the correct directory
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, '../README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="blast_embedding",  
    version="0.1.0",
    description="A BLAST-like algorithm using embeddings for bioinformatics",
    long_description=long_description,  # Use the correct path for README.md
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/embedded-blast",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "biopython",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
