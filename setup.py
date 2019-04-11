import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="basic_ems",
    version="0.0.1",
    author="Cycle ML Team",
    author_email="ml@cyclesystem.org",
    description="A simple EMS based on rules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cycle-system/simple_ems",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
	  'scipy',
	  'scikit-learn==0.19'	
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
