from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

setup(
    name="detect-malignant",
    version=__version__,
    description="detect malignant ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sameerah Talafha",
    author_email="sameerah.talafha@siu.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",    
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "timm",        
        "tensorboard",
        "seaborn",
        "IPython",       
        "black",
        "isort",
        "albumentations",
        "torchsampler",
        "pretrainedmodels",
        "fastai",
        "torchsummary",
        "torch >= 1.4",
        "torchvision",
                ],
    python_requires=">=3.10",
    entry_points={"console_scripts": ["detect-malignant = detect_malignant.__main__:run"]},
)