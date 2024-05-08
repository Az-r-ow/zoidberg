from setuptools import setup, find_packages, Extension
import os, shutil
from utils.files import find_file

with open('requirements.txt') as f:
    required = f.read().splitlines()

neuralNet_build_folder = "libs/NeuralNet/build"
utils_folder = "./utils"
dir_path = os.path.dirname(os.path.realpath(__file__))
neuralNet_so_file = find_file(".so", utils_folder)

# Check for neural net so file otherwise compile it 
if not os.path.exists(neuralNet_build_folder) or not neuralNet_so_file:
  os.chdir('libs/NeuralNet')
  os.system('git submodule init && git submodule update')
  os.system('source ./scripts/build_without_tests.sh')
  os.chdir(dir_path)
  neuralNet_so_file = find_file(".so", neuralNet_build_folder)
  shutil.move(neuralNet_so_file, utils_folder)

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