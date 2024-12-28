# Does Catastrophic Forgetting Happen in Tiny Subspaces?

## Table of Contents
1. [Contributors](#contributors)
2. [Setup](#setup)
3. [Source Code Structure](#source-code-structure)
4. [Licenses](#licenses)

## Contributors
- Rufat Asadli (22-953-632)
- Armin Begic (20-614-582)
- Jan Schlegel (19-747-096)
- Philemon Thalmann (18-111-674)

## Setup
### Installation
```bash
conda create -n cf python=3.10 pip
conda activate cf
pip install -r requirements.txt
```

We moreover make use of the "hessian_eigenthings" package for computing the top-k eigenvectors of the Hessian in a scalable way. "hessian_eigenthings" can be installed by running
```bash
pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
```

This repository includes the "hessian_eigenthings" as a submodule; after cloning, run the following commands at the root level of the main repository. 
```bash 
git submodule init 
git submodule update
```
to initialize and update the submodule.

## Source Code Structure
```bash
├── configs
│   ├── cnn.yaml
│   └── mlp.yaml
├── environment.yaml
├── eval
│   └── __init__.py
├── LICENSE
├── modules
│   ├── cnn.py
│   ├── __init__.py
│   ├── mlp.py
│   └── sgd.py
├── notebooks
│   └── experiments.ipynb
├── README.md
├── reports
│   └── proposal.pdf
├── scripts
│   ├── eval.sh
│   └── train.sh
├── train
│   └── __init__.py
└── utils
    ├── datasets.py
    ├── __init__.py
    ├── metrics.py
    └── plotting.py
```

## Licenses
The file `utils/hvp_operator.py` is an adapted version of the implementation of the Hessian-vector product operator from the "hessian_eigenthings" package. The original implementation can be found [here](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/master/hessian_eigenthings/hvp_operator.py). The corresponding licence that is used for this file is a copy of the original license of the "hessian_eigenthings" package and is named `LICENSE-hessian-eigenthings`. For details about the adaptions we made to the original implementation, we refer to the comment header of the file `pytorch_hessian_eigenthings/hessian_eigenthings/hvp_operator.py`.
