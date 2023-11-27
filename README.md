# OMLT_CAMD

This repository is the official implementation of the paper "Augmenting optimization-based molecular design with graph neural networks".

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

A license is needed to use *Gurobi*. Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). 

## Data preparation

The dataset comes from the paper ["SMILES to Smell: Decoding the Structure–Odor Relationship of Chemical Compounds Using the Deep Neural Network Approach"](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01288) and can be found in the supporting information of this paper. We already download it here. 

To preprocess the raw dataset, run this command (using banana odor as an example):

```
python data_preparation_banana.py
```


## Model training

To train a GNN on the preprocessed dataset, run this command (using banana odor as an example):

```
python model_training_banana.py
```


## Optimization

To solve the optimization problem, run this command (using banana odor as an example):

```
python optimization_banana.py $N $seed_gnn $seed_grb
```
where N is the number of fragments, seed_gnn is the random seed for training GNN, and seed_grb is the random seed of Gurobi.

## Visualize molecules

The solutions found by solving the optimization problems consist of fragment features, adjacency matrix, and double bond matrix. To visualize the molecules corresponding to these solutions, run this command:

```
python plot_molecules.py $odor $N $seed_gnn $seed_grb
```
where odor is the target odor (banana or garlic), N is the number of fragments, seed_gnn is the random seed for training GNN, and seed_grb is the random seed of Gurobi.


# Contributors
Shiqiang Zhang. Funded by an Imperial College Hans Rausing PhD Scholarship.
