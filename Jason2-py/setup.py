import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
def README():
    with open('README.md') as f:
        return f.read()

setuptools.setup(
    name="JASON2_py", # Replace with your own username
    version="0.0.1",
    author="Theo Yang",
    author_email="tsyang@caltech.edu",
    description="Modeling the two-dimensional fourier transform of internal waves using JASON-2 ssha",
    long_description=README(),
    long_description_content_type="text/markdown",
    url="https://github.com/theo-yang/jason2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)