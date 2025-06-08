
<h1 align="center">
  Deep Learning Specialization (DeepLearning.AI)
</h1>
<br/>

### Table of contents
- [Curriculum](https://github.com/jmcheon/deep_learning_specialization/wiki/Curriculum)
  - [Course1 - Neural Networks and Deep Learning](https://github.com/jmcheon/deep_learning_specialization/tree/main/Course1) (4 Modules)
  - [Course2 - Improving Deep Neural Networks](https://github.com/jmcheon/deep_learning_specialization/tree/main/Course2) (3 Modules)
  - [Course3 - Structuring your Machine Learning project](https://github.com/jmcheon/deep_learning_specialization/tree/main/Course3) (2 Modules)
  - [Course4 - Convolutional Neural Networks](https://github.com/jmcheon/deep_learning_specialization/tree/main/Course4) (4 Modules)
  - [Course5 - Natural Language Processing](https://github.com/jmcheon/deep_learning_specialization/tree/main/Course5) (4 Modules)
- [Standard notations for Deep Learning](#standard-notations-for-deep-learning)
	- [1. Neural Networks Notations](#1-neural-networks-notations) 
	- [2. Deep Learning representations](#2-deep-learning-representations)

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


##### Example: 

$$x  \in \mathbb{R}, \space y \in \{0, 1\}$$ 

$m$ - the number of examples: $(x^{(1)}, y^{(1)}), (x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})$

$$X=
\left[
\begin{matrix}
| & | & | & | & |\\
x^{(1)} & x^{(2)} & x^{(3)} & \dots & x^{(m)} \\
| & | & | & | & | \\
\end{matrix}
\right], \space
Y=
\left[
\begin{matrix}
y^{(1)} & y^{(2)} & y^{(3)} & \dots & y^{(m)}
\end{matrix}
\right]
$$

$$X  \in \mathbb{R}^{n_{x} \times m }, \space X.shape = (n_x, m)$$ 

$$Y \in \mathbb{R}^{1 \times m }, \space Y.shape = (1, m)$$

#### Common forward propagation equation examples:
- $a = g^{[l]} (W_x x^{(i)} + b_1) = g^{[l]} (z_1)$ where $g^{[l]}$ denotes the $l^{th}$ layer activation function
- $\hat y^{(i)} = softmax(W_h h + b_2)$
- General Activation Formula: $a_{j}^{[l]} = g^{[l]} ( \sum_{k} w^{[l]}_{jk} a^{[l−1]}_k + b^{[l]}_j ) = g^{[l]} (z^{[l]}_j )$
- $J(x, W, b, y)$ or $J(\hat y, y)$ denote the cost function

#### Examples of cost function:
$$J_{CE}(\hat y, y) = − \sum^m_{i=0} y^{(i)} \log \hat y^{(i)}$$

$$J_1(\hat y, y) = \sum^{m}_{i=0} | y^{(i)} − \hat y^{(i)} |$$

### 2. Deep Learning representations
- nodes represent inputs, actiavtions and outputs
- edges represent weights and biases

Here are several examples of standard deep learning representations

| Comprehensive Network | Simplified Network |   
| :------: | :------------------------: |
| <img alt="DL representation1" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/8763f9d7-01f7-4517-81d1-9d663bcfacee" width=500 height=250>| <img alt="DL representation2" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/7a10dd58-f758-4adf-92c7-302b3bd3dfe9" width=500 height=250 >|
