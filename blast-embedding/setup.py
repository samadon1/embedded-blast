from setuptools import setup, find_packages

setup(
    name="blast_embedding",  # The package name
    version="0.1.0",  # Version number
    description="A BLAST-like algorithm using embeddings for bioinformatics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",  # Specifies markdown for the long description
    author="Alexander Titus",  # Replace with your name
    author_email="your-email@example.com",  # Replace with your email
    url="https://github.com/In-Vivo-Group/embedded-blast",  # URL of the project repository
    packages=find_packages(where="src"),  # Looks for packages inside the `src` directory
    package_dir={"": "src"},  # Tells setuptools that packages are under `src/`
    install_requires=[  # Add your package dependencies here
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "biopython",
    ],
    classifiers=[  # Metadata for your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Minimum Python version required
)
