import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepquest",
    version="0.0.1",
    author="David Wilby",
    author_email="d.wilby@sheffield.ac.uk",
    description="A framework for neural-based quality estimation for machine translation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RSE-Sheffield/deepquest/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "future",
        "numpy<1.17,>=1.14.5",
        "scipy",
        "pyyaml",
        "tensorflow==1.14.0",
        "multimodal_keras_wrapper",
        "Theano",
        "keras @ https://github.com/MarcBS/keras/archive/master.zip",
        "nmt_keras @ https://github.com/davidwilby/nmt-keras/archive/setup.zip"
    ],
    packages=setuptools.find_packages(),
    package_data={
        'deepquest':['deepquest/configs/*']
    },
    python_requires='>=3.6',
    entry_points={
          'console_scripts': [
              'dq = deepquest.__main__:main',
              'deepquest = deepquest.__main__:main'
          ]
      },
)
