import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="earwole1",
    version="0.0.1",
    author="earwole1",
    author_email="earwole1@gmail.com",
    description="A simple covid tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earwole1/covid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
