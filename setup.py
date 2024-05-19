from setuptools import setup, find_packages

setup(
    name='NeuralNetworkLibrary',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0'      
    ],
    author='Jesus Daniel Gonzalez Rocha',
    author_email='daniel.gonzalez13@uabc.edu.mx',
    description='Libreria para la implementacion de redes neuronales densas',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Dan1543/Neural-Network',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)