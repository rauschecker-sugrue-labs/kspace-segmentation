# 🌌 kseg

K-space segmentation models

<p float="left">
  <a href="https://www.pytorch.org/" title="PyTorch"><img src="https://github.com/get-icon/geticon/raw/master/icons/pytorch.svg" alt="PyTorch" width="21px" height="21px"></a>
 <a href="https://www.lightning.ai/" title="TorchIO"><img src="https://torchio.readthedocs.io/_static/torchio_logo_2048x2048.png" alt="TorchIO" width="21px" height="21px"></a>
    <a href="https://github.com/fepegar/torchio" title="Lightning AI"><img src="https://avatars.githubusercontent.com/u/58386951?s=200&v=4" alt="Lightning AI" width="21px" height="21px"></a>
  <a href="https://docs.ray.io/en/latest/tune/index.html" title="Ray Tune"><img src="https://avatars.githubusercontent.com/u/22125274?s=200&v=4" alt="Ray Tune" width="21px" height="21px"></a>
</p>

## Installation


Clone the repo
```bash
git clone git@git.ucsf.edu:rauschecker-sugrue-labs/kspace-segmentation.git
cd kspace-segmentation
```

Then run (we recommend using a python virtual environment)

```bash
pip install --upgrade pip
pip install --upgrade setuptools
pip install -e .
```

## Training
Run

```bash
kseg train [model] [data]
```

## Create documentation
To generate the documentation, make sure sphinx is installed and run

```bash
cd docs
make html
```

If you added a new module to the kseg project, you have to update the .rst files in the docs/source/ directory. You can use

```bash
cd kspace-segmentation
sphinx-apidoc -F -e -o docs/source kseg
```
