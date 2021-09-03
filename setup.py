"""Package Setup"""
import os
import re
from distutils.core import setup

from setuptools import find_packages

CURRENT_DIR = os.path.dirname(__file__)


def read(path):
    with open(path, "r") as filep:
        return filep.read()


def get_version(package_name):
    with open(os.path.join(os.path.dirname(__file__), package_name, "__init__.py")) as fp:
        for line in fp:
            tokens = re.search(r'^\s*__version__\s*=\s*"(.+)"\s*$', line)
            if tokens:
                return tokens.group(1)
    raise RuntimeError("Unable to find own __version__ string")


setup(
    name="lorien",
    version=get_version("lorien"),
    license="Apache-2.0",
    description="A Unified Infrastructure for Efficient Deep Learning Workloads Delivery",
    long_description=read(os.path.join(CURRENT_DIR, "README.md")),
    long_description_content_type="text/markdown",
    author="Lorien Community",
    url="https://github.com/awslabs/lorien",
    keywords=[],
    packages=find_packages(),
    install_requires=[p for p in read("requirements.txt").split("\n") if p],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
