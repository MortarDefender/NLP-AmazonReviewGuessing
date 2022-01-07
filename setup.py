import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\n")


setup(
    name = 'Amazon Review Classifiew',
    version= "1.0.1",
    description='Amazon Review Classifiew for python',
    long_description='classification of a review from amazon on 1 start to 5 star rateing',
    author='Mortar Defender',
    license='MIT License',
    url = '__',
    setup_requires = getRequirements(),
    install_requires = getRequirements(),
    include_package_data=True
)
