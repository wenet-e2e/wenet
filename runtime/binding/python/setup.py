import datetime
import os
import re
import subprocess

from setuptools import find_packages, setup

install_requires=[
    "tqdm",
    "requests",
]

setup(
    name='wenet',
    version="0.0.1",
    description='wenet runtime python binding',
    install_requires=install_requires,
    long_description_content_type='text/markdown',
    author='Mddct',
    # keywords='speech recognition',
    packages=['wenet'],
    package_data={"wenet":["lib/*"]},
    zip_safe=False,
)
