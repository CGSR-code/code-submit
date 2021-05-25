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
We refer to:
 - [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN) for the proprocess.py in our repository
 - [FGNN](https://github.com/RuihongQiu/FGNN) for the WGAT

## Appendix
- Causality and correlation statistics on Gowalla and Amazon
  <img src="pics/Causality and correlation statistics.png" align="center" width="75%" style="margin: 0 auto">
- Hyper-parameter setups of baselines
  | Method  | Datasets                            | Hyper-parameter setups                                       |
  | ------- | ----------------------------------- | :----------------------------------------------------------- |
  | GRU4Rec | Diginetica,Gowalla,Amazon           | GRU size=100, Batch size=32, Learning rate=0.2               |
  | NARM    | Diginetica,Gowalla,Amazon           | Embedding size=50, Batch size=512, Learning rate=0.001       |
  | SR-GNN  | Diginetica, Gowalla<br />Amazon     | Embedding size=100, Batch size=100, Learning rate=0.001, $L_2$ penalty=1e-5<br />Embedding size=170, Batch size=100, Learning rate=0.001, $L_2$ penalty=1e-5 |
  | FGNN    | Diginetica, Gowalla<br />Amazon     | Embedding size=100, Batch size=100, Learning rate=0.001, $L_2$ penalty=1e-5<br />Embedding size=150, Batch size=100, Learning rate=0.001, $L_2$ penalty=1e-5 |
  | LESSR   | Diginetica<br />Gowalla<br />Amazon | Embedding size=32, Batch size=512, Learning rate=0.001, $L_2$ penalty=1e-4<br />Embedding size=64, Batch size=512, Learning rate=0.001, $L_2$ penalty=1e-4<br />Embedding size=128, Batch size=512, Learning rate=0.001, $L_2$ penalty=1e-4 |
