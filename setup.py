from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='s2wrapper',
    version='0.1',
    author='Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, Trevor Darrell',
    author_email='baifeng_shi@berkeley.edu',
    description='Pytorch implementation of S2Wrapper, a simple mechanism to extract multi-scale features using any vision model. Please refer to paper:',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bfshi/s2',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'einops'
    ],
)