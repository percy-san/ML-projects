from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements required for the package.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        "Removed the -e . from the requirements list"
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='End-End ML Project',
    version='1.0',
    author='Buhlebethu Mkhonta',
    author_email='buhlebethumkhonta@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)
