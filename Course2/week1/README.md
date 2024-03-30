## week1 - Practical Aspects of Deep Learning

### Table of contents
1. [Regularizing your Neural Network](#1)
	- 1-1. [Regularization](#1-1)
	- 1-2. [Dropout Regularization](#1-2)
	- 1-3. [Other Regularization methods](#1-3)
2. [Setting up your Optimization Problem](#2)

<a id="1"></a>
### 1. Regularizing your Neural Network
<a id="1-1"></a>
#### 1-1. Regularization
#### keywords
- regularization parameter $\lambda$
- L1, L2 regularization
- Forbenius norm
- bias, variance

#### keypoints
- The intuition of completely zeroing out a bunch of hidden units
- High bias vs high variance cases

#### questions
- why regularization reduces overfitting?/ why does it help variance problem?
- how does regularization prevent overfitting?

<a id="1-2"></a>
#### 1-2. Dropout Regularization
#### keywords
- inverted dropout
- diminished network
- spreading out weights
- effect of shrinking the squared norm of the weights

#### keypoints
- zeroing out different hidden units during the iteration
- randomly knocking out units in the network

#### questions
- what does it mean to zero out different hidden units?
- when to use dropout regluarization?

<a id="1-3"></a>
#### 1-3. Other Regularization methods
#### keywords
- data augmentation
- early stopping

#### questions
- what is the downside of early stopping?

<a id="2"></a>
### 2. Setting up your Optimization Problem
- Normalizing inputs
- Vanishing / Exploding gradients
- Weight initialization for deep networks
- Numerical approximation of gradients
- Gradient checking
