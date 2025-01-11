# Does Catastrophic Forgetting Happen in Tiny Subspaces?
Catastrophic forgetting remains a significant challenge in continual learning, where adapting to new tasks often disrupts previously acquired knowledge. Recent studies on neural network optimization indicate that learning in a non-continual framework primarily occurs within the bulk subspace of the loss Hessian, which is associated with small eigenvalues of the latter. However, the role of the bulk subspace in a continual learning setting, particularly in relation to forgetting, is not well understood. In this work, we investigate how constraining gradient updates to either the bulk or dominant subspace affects learning and forgetting. Through experiments on Permuted MNIST, Split-CIFAR10, and Split-CIFAR100, we confirm that task-specific learning occurs in the bulk subspace of the loss Hessian. Additionally, there is evidence suggesting that forgetting may also predominantly occur within the bulk subspace, although further large-scale experiments are needed to validate this. Our findings provide promising avenues for efficient implementations of algorithms that counter catastrophic forgetting.
## Table of Contents

1. [Contributors](#contributors)
2. [Setup](#setup)
3. [Source Code Structure](#source-code-structure)
4. [Reproducibility](#reproducibility)

## Contributors

- Rufat Asadli (22-953-632)
- Armin Begic (20-614-582)
- Jan Schlegel (19-747-096)
- Philemon Thalmann (18-111-674)

## Setup

### Installation

Clone this repository.
```bash
git clone git@github.com:JHSchlegel/cf-tiny-subspaces.git
cd cf-tiny-subspaces
```

We recommend using a conda environment to install the required packages. Run the following commands to create a new environment and install the dependencies.
```bash
conda create -n cf python=3.10 pip
conda activate cf
pip install -r requirements.txt
```

Moreover, this repository includes [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings/tree/master) as a submodule; after cloning, run the following commands at the root level of the `cf-tiny-subspaces` repository to initialize, update and install the submodule:

```bash
git submodule init 
git submodule update
cd pytorch_hessian_eigenthings
pip install -e .
```

## Source Code Structure

```bash
.
├── LICENSE                  
├── README.md
├── requirements.txt
├── reports
│   ├── proposal.pdf                        # project proposal
│   └── paper.pdf                           # final project report
├── configs
│   ├── permuted_mnist.yaml                 # configuration file for permuted MNIST
│   ├── split_cifar10.yaml                  # configuration file for split CIFAR-10
│   └── split_cifar100.yaml                 # configuration file for split CIFAR-100
├── modules
│   ├── __init__.py
│   ├── CLTrainer.py                        # trainer for continual learning
│   ├── JointTrainer.py                     # trainer for multitask learning
│   ├── cnn.py                              # CNN model for split CIFAR-10 and split CIFAR-100
│   ├── mlp.py                              # MLP model for permuted MNIST
│   └── subspace_sgd.py                     # SGD-optimizer that allows for gradient projection into a supspace
├── notebooks
│   ├── multitask_learning.ipynb            # notebook for multitask learning
│   ├── subspace_sgd_examples.ipynb         # tutorial notebook for subspace-SGD
│   └── visualizations.ipynb                # notebook for visualizations
├── scripts
│   ├── ablation_study.sh                   # script to run bulk space ablations
│   └── train.sh                            # script to run main experiments
├── train
│   ├── __init__.py
│   ├── train_permuted_mnist.py             # train script for permuted MNIST
│   ├── train_split_cifar10.py              # train script for split CIFAR-10
│   └── train_split_cifar100.py             # train script for split CIFAR-100
└── utils
    ├── data_utils
    │   ├── continual_dataset.py            # abstract dataset class for continual learning
    │   ├── permuted_mnist.py               # permuted MNIST dataset
    │   └── sequential_CIFAR.py             # sequential CIFAR datasets
    ├── __init__.py
    ├── metrics.py                          # overlap metrics
    ├── reproducibility.py                  # utilities for pytorch reproducibility
    └── wandb_utils.py                      # utilities for logging to wandb
```

## Reproducibility

The main results of our work, as summarized in `Table 1` of the report, can be reproduced by running the `scripts/train.sh` script:
```bash
bash scripts/train.sh
```  

The bulk space ablations can be reproduced by running the `scripts/ablation_study.sh` script:
```bash
bash scripts/ablation_study.sh <dataset_name>
```
where `<dataset_name>` is one of `cifar10`, `cifar100`, or `pmnist`.


All visualizations included in the report were created using the `notebooks/visualizations.ipynb` notebook.

Finally, the multitask oracle baseline can be reproduced by running the `notebooks/multitask_learning.ipynb` notebook.
