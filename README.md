## Requirements
The code has been tested under Python 3.7.4. The required packages are as follows:
- pytorch 1.7.0
- numpy 1.19.2
- torch-geometric 1.6.1

## Datasets
You can download experiment data via following links, and then put them into the "datasets" folder:
  - [Diginetica](https://competitions.codalab.org/competitions/11161)
  - [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
  - [Amazon Home and Kitchen](http://jmcauley.ucsd.edu/data/amazon/links.html)

## Example to Run the Codes
Train and evaluate the model:
```
python build_relation_graph.py --dataset=diginetica
python main.py --dataset=diginetica --hidden_size=110 --batch_size=20 --reg=1e-6 --dropout=0.6
```

## Baselines
The codes for baselines can be found at:
  - [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN)
  - [FGNN](https://github.com/RuihongQiu/FGNN)
  - [LESSR](https://github.com/twchen/lessr)

## Acknowledgements
The code for data preprocessing can refer to [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).
