from setuptools import setup, find_packages

requirements = [
    "tqdm",
    "torch==1.13.0",
    "torchaudio==0.13.0"
]

setup(
    name="wenet",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wenet = wenet.cli.transcribe:main",
    ]},
)
