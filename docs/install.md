# Installation

jnkepler requires Python 3.10 or later. Use a virtual environment to keep the
analysis dependencies separate from other projects.

## Install from PyPI

```bash
pip install jnkepler
```

The TTV tutorials use dependencies installed with jnkepler and a data file
included in the package. For photodynamical modeling, also install `jaxoplanet`;
the GP likelihood additionally needs `tinygp`. See
[Models and extensions](models.md) for the prediction interfaces and
[Examples](examples.md) for notebook-specific requirements.

Before importing JAX or jnkepler, apply the CPU settings shown at the start of
[part 1](quickstart.md#set-up-the-session). In a notebook where JAX has already
been imported, restart the kernel first. The
[README](https://github.com/kemasuda/jnkepler#installation) discusses
version-specific performance settings in more detail.

## Development install

```bash
git clone https://github.com/kemasuda/jnkepler.git
cd jnkepler
pip install -e .
```

## Build the documentation locally

From the repository root, with the package installed:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

Open `docs/_build/html/index.html`. Building HTML does not execute the tutorial
code or run the long example notebooks. It uses the checked-in API pages;
`make -C docs apidoc` explicitly regenerates those pages when needed.
