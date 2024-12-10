# Does Catastrophic Forgetting Happen in Tiny Subspaces?

## Table of Contents
1. [Contributors](#contributors)
2. [Setup](#setup)
3. [Source Code Structure](#source-code-structure)

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
