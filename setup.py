"""
Package Setup Configuration

This module configures the Python package for distribution and installation.
It defines package metadata and dependencies required for the project.
"""

from setuptools import setup, find_packages
from typing import List

# Constant to identify editable installs in requirements.txt
HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements from a file and return them as a list.
    
    This function reads the requirements.txt file and processes each line
    to create a clean list of package dependencies. It also removes
    the editable install marker (-e .) from the requirements list.
    
    Args:
        file_path (str): Path to the requirements file (usually 'requirements.txt')
    
    Returns:
        List[str]: List of package requirements without newlines and without -e .
    
    Example:
        If requirements.txt contains:
        numpy==1.21.0
        pandas==1.3.0
        -e .
        
        This function returns: ['numpy==1.21.0', 'pandas==1.3.0']
    """
    requirements = []
    
    # Open and read the requirements file
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        # Remove newline characters from each requirement
        requirements = [req.replace('\n', '') for req in requirements]

        # Remove the editable install marker (-e .) from the requirements list
        # This is used for local development but shouldn't be in the final package
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Configure the package setup with metadata and dependencies
setup(
    name='End-End ML Project',  # Name of the package
    version='1.0',              # Version number
    author='Buhlebethu Mkhonta',  # Author name
    author_email='buhlebethumkhonta@gmail.com',  # Author email
    packages=find_packages(),    # Automatically find all Python packages in the project
    install_requires=get_requirements('requirements.txt'),  # List of required dependencies
)
