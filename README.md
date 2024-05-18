# Zoidberg

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
