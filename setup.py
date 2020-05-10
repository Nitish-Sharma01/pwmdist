import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'pwmdist'
AUTHOR = 'Nitish Sharma'
AUTHOR_EMAIL = 'nitish.sharma0712@gmail.com'
URL = 'https://github.com/Nitish-Sharma01/pwmdist'

LICENSE = 'MIT License'
DESCRIPTION = 'package for essential statistics of extreme value distirbutions using probability weighted moments'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'scipy',
      'matplotlib',
      'seaborn'
]
CLASSIFIERS = [
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
]
PYTHON_REQUIRES = '>=3.5'
setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      +find_packages(where="./pwmdist"),
      classifiers=CLASSIFIERS
      )
