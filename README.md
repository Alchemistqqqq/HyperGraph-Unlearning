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
python influence_function.py --dataset Cooking200
python delete.py --dataset Cooking200
python HSCD.py --dataset Cooking200
python partition.py --dataset Cooking200
dataset:"Cooking200", "CoauthorshipCora","CoauthorshipDBLP","CocitationCora",
"CocitationCiteseer","Recipe100k","Recipe200k",
```

* Second, features can be generated using the data set itself or using the identity matrix:
```
X, lbl = torch.eye(data["num_vertices"]).to(device), data["labels"].to(device)
or
X = data["features"].to(device)
lbl = data["labels"].to(device)
```

* Next, to obtain the original model embedding:
```
cd attack
python original.py
```

* Last, to conduct the MIA experiment:
```  
cd edge/node/feature
python retrain.py --dataset Cooking200
python if.py --dataset Cooking200
python delete.py --dataset Cooking200
python HSCD.py --dataset Cooking200
python partition.py --dataset Cooking200
```


