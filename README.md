# Zoidberg

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
  

## Quick Start

Create a python virtual environment :

```bash
python3 -m venv venv
```

Activate the virtual env :

```bash
source ./venv/bin/activate
```

Install the requirements

```bash
python3 -m pip install -r requirements.txt
```

Build the project with :

```bash
python3 setup.py install
```

If you get an error of the sort :

```
ModuleNotFoundError: No module named 'setuptools'
```

Install `setuptools`

```bash
python3 -m pip install -U pip setuptools
```

## Project structure 

The main entrypoint to the project is : `main.ipynb` 

However the second iteration (an improvement from `main.ipynb` is `revised_main.ipynb`.

If you're interested in the preprocessing steps these will be in the `/src` folder. 
