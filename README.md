# OMLT_CAMD

This repository is the official implementation of the paper ["Augmenting optimization-based molecular design with graph neural networks"](https://doi.org/10.1016/j.compchemeng.2024.108684). This paper was publised in Computers \& Chemical Engineering. Please cite as:

- Shiqiang Zhang, Juan S. Campos, Christian Feldmann, Frederik Sandfort, Miriam Mathea, Ruth Misener. "Augmenting optimization-based molecular design with graph neural networks." Computers \& Chemical Engineering 186 (2024): 108684.

The BibTex reference is:

     @article{zhang2024,
          title = {Augmenting optimization-based molecular design with graph neural networks},
          author= {Shiqiang Zhang and Juan S. Campos and Christian Feldmann and Frederik Sandfort and Miriam Mathea and Ruth Misener},
          journal = {Computers \& Chemical Engineering},
          volume = {186},
          pages = {108684},
          year = {2024},
          issn = {0098-1354},
          doi = {https://doi.org/10.1016/j.compchemeng.2024.108684},
     }


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
python model_training_banana.py $seed_gnn
```
where seed_gnn is the random seed for training GNN.

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
[*Shiqiang Zhang*](https://github.com/zshiqiang). Funded by an Imperial College Hans Rausing PhD Scholarship.
