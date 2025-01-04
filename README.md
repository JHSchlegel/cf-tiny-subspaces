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

This repository includes the "hessian_eigenthings" as a submodule; after cloning, run the following commands at the root level of the main repository. 
```bash 
git submodule init 
git submodule update
cd pytorch_hessian_eigenthings
pip install -e .
```
to initialize, update and install the submodule.

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
