# PyQumo - Queueing models in Python

## How to install for development

1. Clone the repository (say, to `/home/username/pyqumo`) and go to its root:

```bash
~> git clone https://github.com/ipu69/pyqumo
```

2. Create a virtual environment using Python 3.8 and activate it.
To display a nice message when using the venv you can provide its name using `--prompt` key:

```bash
~/pyqumo> python3.8 -m venv .venv --prompt pyqumo
~/pyqumo> source .venv/bin/activate
```

3. Install the package in development mode:

```bash
~/pyquom> pip install -e .
```

4. Run tests:

```bash
~/pyqumo> pip install pytest
~/pyqumo> pytest
```
