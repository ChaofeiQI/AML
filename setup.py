#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="AML",
    version=1.0,
    author="Chaofei",
    url="https://github.com/ChaofeiQI/AML",
    description="Codebase for few-shot object detection",
    python_requires=">=3.8",
    packages=find_packages(exclude=('configs', 'data', 'work_dirs')),
    install_requires=[
        'clip@git+ssh://git@github.com/openai/CLIP.git'
    ],
)