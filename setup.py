from setuptools import setup, find_packages
import os 

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
  name='zoidberg',
  version='1.0.0',
  author='Antoine Azar',
  author_email='antoine.azar123@gmail.com',
  description='An epitech project',
  packages=find_packages(),
  install_requires=required,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)