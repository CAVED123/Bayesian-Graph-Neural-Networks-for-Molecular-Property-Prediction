from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chempropBayes',
    version='0.0.1',
    author='William Lamb',
    author_email='wglamb196@gmail.com',
    description='Bayesian Molecular Property Prediction with Message Passing Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/georgelamb19/chempropBayes',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'Bayesian',
        'property prediction',
        'message passing neural network',
        'graph neural network'
    ]
)
