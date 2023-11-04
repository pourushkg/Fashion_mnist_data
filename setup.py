from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.2",
    author="pourush",
    description="A small package for ANN Implementation for fashion_mnist dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pourushkg/Fashion_mnist_data.git",
    author_email="pkg.21p10161@mtech.nitdgp.ac.in",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)