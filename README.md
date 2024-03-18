
<h1 align="center">
  Deep Learning Specialization
</h1>
<br/>

### Table of contents
- [Curriculum](https://github.com/jmcheon/deep_learning_specialization/wiki/Curriculum)
  - [Module1 - Neural Networks and Deep Learning](https://github.com/jmcheon/deep_learning_specialization/wiki/Module1-%E2%80%90-Neural-Networks-and-Deep-Learning) (4 weeks)
  - [Module2 - Improving Deep Neural Networks](https://github.com/jmcheon/deep_learning_specialization/wiki/Module2-%E2%80%90-Improving-Deep-Neural-Networks) (3 weeks)
  - [Module3 - Structuring your Machine Learning project](https://github.com/jmcheon/deep_learning_specialization/wiki/Module3-%E2%80%90-Structuring-your-Machine-Learning-project) (2 weeks)
  - [Module4 - Convolutional Neural Networks](https://github.com/jmcheon/deep_learning_specialization/wiki/Module4-%E2%80%90-Convolutional-Neural-Networks) (4 weeks)
  - [Module5 - Natural Language Processing](https://github.com/jmcheon/deep_learning_specialization/wiki/Module5-%E2%80%90-Natural-Language-Processing) (4 weeks)
- [Standard notations for Deep Learning](#standard-notations-for-deep-learning)
	- [Neural Networks Notations](#neural-networks-notations)

## Standard notations for Deep Learning
### 1. Neural Networks Notations
#### General comments: 
- superscript $(i)$ will denote the $i^{th}$ training example
- superscript $[l]$ will denote the $l^{th}$ layer

#### Sizes:
- $m$ : number of examples in the dataset
- $n_x$ : input size 
- $n_y$ : output size (or number of classes) 
- $n^{[l]}_h$ : number of hidden units of the $l^{th}$ layer
- In a for loop, it is possible to denote $n_x = n^{[0]}_h$ and $n_y = n_h^{[\text {number of layers} +1]}$ 
- $L$ : number of layers in the network

#### Objects:
- $X  \in \mathbb{R}^{n_{x} \times m }$ is the input matrix 
- $x^{(i)}  \in \mathbb{R}^{n_{x}}$ is the $i^{th}$ example represented as a column vector
- $Y \in \mathbb{R}^{n_y \times m }$ is the label matrix 
- $y^{(i)}  \in \mathbb{R}^{n_{y}}$ is the output label for the $i^{th}$ example 
- $W^{[l]}  \in \mathbb{R}^{ \text{number of units in next layer} \times \text{number of units in the previous layer}}$ is the weight matrix, superscript $[l]$ indicates the layer 
- $b^{[l]}  \in \mathbb{R}^{ \text{number of units in next layer}}$ is the bias vector in the $l^{th}$ layer 
- $\hat y^{(i)}  \in \mathbb{R}^{n_{y}}$ is the predicted output vector. It can also be denoted $a ^{[L]}$ where $L$ is the number of layers in the network

#### Common forward propagation equation examples:
- $a = g^{[l]} (W_x x^{(i)} + b_1) = g^{[l]} (z_1)$ where $g^{[l]}$ denotes the $l^{th}$ layer activation function
- $\hat y^{(i)} = softmax(W_h h + b_2)$
- General Activation Formula: $a_{j}^{[l]} = g^{[l]} ( \sum_{k} w^{[l]}_{jk} a^{[l−1]}_k + b^{[l]}_j ) = g^{[l]} (z^{[l]}_j )$
- $J(x, W, b, y)$ or $J(\hat y, y)$ denote the cost function

#### Examples of cost function:
$$J_{CE}(\hat y, y) = − \sum^m_{i=0} y^{(i)} \log \hat y^{(i)}$$

$$J_1(\hat y, y) = \sum^{m}_{i=0} | y^{(i)} − \hat y^{(i)} |$$
