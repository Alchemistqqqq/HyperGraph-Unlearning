# HyperGraph-Unlearning
This code serves as a code for Hypergraph Unlearning, which includes hyperedge unlearning, node unlearning, and feature unlearning. 

## Environment
```
conda create -n hyper python=3.9
conda activate hyper
pip install dhg
pip install numpy==1.26.0 
pip install pure_eval
pip install psutil
```

DHG library specific use tutorial： https://github.com/iMoonLab/DeepHypergraph

## Running examples
```
cd HyperGraph-Unlearning
```
* First, to train the unlearning moedl and test:
```
cd edge/node/feature
python retrain.py --dataset Cooking200
```
