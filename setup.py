from setuptools import setup, find_packages

requirements = [
    "torch==1.10.0",
    "torchaudio==0.10.0"
]

setup(
    name="wenet",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wenet = wenet.cli.transcribe:main",
    ]},
)
