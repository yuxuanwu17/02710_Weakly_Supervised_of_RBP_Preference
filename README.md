# WeakRBP: Weakly supervised learning of RNA-protein binding preference prediction

## required library

- RNAfold
- Forgi
- PyTorch
- scikit-learn
- Numpy
- Pandas

## organized_to_submit directory


### deep learning model file

- main.ipynb
- model.py
- dataloader.py

### RNA secondary structure

- rnaCovert.py
- RNA_structure_generation.py

### Dataset

- dataset can be found here: https://github.com/xypan1232/iDeepS/tree/master/datasets/clip

### Model
![model_sketch](https://cdn.jsdelivr.net/gh/imgstore/typora/20220714144814.png)


## Idea 

In genomics, weakly supervised learning, especially multi-instance learning (MIL), has been intensively applied for studying protein–DNA interaction (Gao and Ruan, 2015, 2017; Zhang et al., 2019, 2020), with the basic assumption that `the sequences captured by CLIP (or ChIP-seq) technologies contain both the interacting and non-interacting elements with the proteins`. We know only the label of the entire sequence, but it is not exactly clear which part of the sequence plays the key role, and a significant proportion of it may not contribute to the binding between DNA and protein at all. 



## Multi-instance learning

- A transformation of instances to a low-dimensional embedding
- A permutation-invariant (symmetric) aggregation function
- A final transformation to the bag probability

## Model explanation

How to obtain bag level probabilities from the instance level features without instance level labels

- MAX: extract information only concerning the most favored instance (overlooks other valuable instances, suffer from the outliers) `MaxPooling`
- Average pooling: assign equal weights to all the instances, ignoring the fact that instances are sparsely distributed. `Average pooling`
- Noisy and method: Not learnable
- Our methods `Gated attention methods`: learnable weights



## Gated attention

- A feature merging approach
- Use a three layer neural network to learn weights $a_k$ of the low dimensional representation of each instance and obtains the bag level embedding according to the equation $Z=\sum_{k=1}^{K}a_kh_k$ where h is a bag of K instance feature 
- A tanh and sigmoid to learn the nonlinearity information 
- Softmax to identify the significant features
- A fully connected layer then takes the element-wise multiplication of two non-linearizes and return the gated attention weights for each instance, where the parameters could be learnable


Attention would measure the degree of similarity among instances and thus is a suitable for our context-dependent data.



