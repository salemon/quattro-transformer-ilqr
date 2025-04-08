import os
from setuptools import setup, find_packages

# Utility function to read the requirements.txt file.
def parse_requirements(filename='requirements.txt'):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        # Read each line, stripping any whitespace characters and ignore comments
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name='quattro_ilqr_tf',
    version='0.1.0',
    packages=find_packages(),
    description='A library that accelerate iLQR using transformer-based models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yue Wang',
    author_email='yue.wang@soton.ac.uk',
    install_requires=parse_requirements(),  # This loads the requirements from requirements.txt
    python_requires='>=3.10',
)