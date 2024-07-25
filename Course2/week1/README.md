
# week1 - Practical Aspects of Deep Learning
### Summary
> Discover and experiment with a variety of different initialization methods, apply L2 regularization and dropout to avoid model overfitting, and then apply gradient checking to identify errors in a fraud detection model.
> - Learning Objectives
>    - Give examples of how different types of initializations can lead to different results
>    - Examine the importance of initialization in complex neural networks
>    - Explain the difference between train/dev/test sets
>    - Diagnose the bias and variance issues in your model
>    - Assess the right time and place for using regularization methods such as dropout or L2 regularization
>    - Explain Vanishing and Exploding gradients and how to deal with them
>    - Use gradient checking to verify the accuracy of your backpropagation implementation
>    - Apply zeros initialization, random initialization, and He initialization
>    - Apply a regularization to a deep learning model

<br>

In this first week, we will first talk about the cellular machine learning problem, then randomization and then some tricks for making sure that your neural network implementation is correct.

### Table of contents
1. [Setting up your Machine Learning Application](#1)
	- 1-1. [Train / Dev / Test sets](#1-1)
	- 1-2. [Bias & Variance](#1-2)
	- 1-3. [Basic Recipe for Machine Learning](#1-3)
2. [Regularizing your Neural Network](#2)
	- 2-1. [Regularization](#2-1)
	- 2-2. [Dropout Regularization](#2-2)
	- 2-3. [Other Regularization methods](#2-3)
3. [Setting up your Optimization Problem](#3)
	- 3-1. [Normalizing inputs](#3-1)
	- 3-2. [Weight initialization for deep networks](#3-2)

<a id="1"></a>
## 1. Setting up your Machine Learning Application
<a id="1-1"></a>
### 1-1.  Train / Dev / Test sets
#### keywords
- cellular machine learning problem
- hold-out cross validation set
- unbiased estimate

#### keypoints
- If the dataset is relatively small, we tend to split it into proportions of 60%, 20%, and 20% for the train, dev, and test sets, respectively. However, if the dataset is large enough, for example, more than 1 million samples, it is acceptable to have less than 1% of the dataset for the dev and test sets.
- Train/dev/test sets come from the same distribution
- The goal of test set is to give you an unbiased estimate of the performance of your final network
- The purpose of having distinct train, dev, and test sets is to allow you to more efficiently measure the bias and variance of your algorithm so that you can more efficiently select ways to improve your algorithm

#### questions
- The purpose of each train, dev, and test set

<a id="1-2"></a>
### 1-2. Bias & Variance trade-off

If you plot the decision boundary, you see if it's underfitting (high bias) or overfitting (high variance).
In high-dimensional problems, there are a couple of different metrics to try to understand the bias and variance.

- Analysis Example

||||||
|--|--|--|--|--|
|Training Set Error|1%|15%|15%|0.5%|
|Dev Set Error|11%|16%|30%|1%|
|Diagnosis|high variance|high bias|high bias &<br> high variance|low bias &<br> low variance|

- Human-level performance
Bayes Error / Optimal Error (Bayesian Optimal Error) $\approx$ 0%


In the case of how to analyze bias and variance, we refer to the Optimal Error compared to the classifier's error to diagnose its bias and variance.
#### keywords
- bias, variance
- Bayes Error / Optimal Error (Bayesian Optimal Error) 

#### keypoints
- You can get a sense of how well you are fitting at least the training data that tells you if you have a bias problem.
- You can get a sense of how bad is the variance problem by looking at how much higher the error goes when you go from the training set to the dev set.
- If your training and dev sets are drawn from the same distribution under the assumption that Bayes error is quite small, you're doing a good job generalizing from a training set to a dev set that gives you a sense of your variance.


<a id="1-3"></a>
### 1-3. Basic Recipe for Machine Learning

|Diagnosis|Basic Recipe to try|
|--|--|
|High bias<br>(training data performance)|- **Bigger network**(more hidden layers, hidden nuits)<br> - Train longer<br> - Advanced optimization algorithm<br> - Appropriate neural network architecture|
|High variance<br>(dev data performance) |- **Get more data**<br> - Regularization(to reduce overfitting)<br> - Appropriate neural network architecture|

#### keypoints
- Use train and dev set to diagnose if you have a bias or variance problem then apply some basic recipes according to your problem

#### questions
- what does it mean to be able to generalize your model?
	- having a good training set performance as well as having a good dev set performance

<a id="2"></a>
## 2. Regularizing your Neural Network
<a id="2-1"></a>
### 2-1. Regularization

In logistic regression:

**L1 Regularization**
$$\frac {\lambda} {m} \sum_{j=0}^{n_x} |w| = \frac {\lambda} {m} ||w||_1 $$

**L2 Regularization** 
Euclidean norm or L2 norm with the parameter vector $w$
$$\frac {\lambda} {2m} ||w||^{2}_2$$

where, $$||w||^{2}_{2}  =\sum^{n_x} _{j=0}  w^{2} _{j} = W^T \cdot W$$

In a neural network:

**Forbenius norm**
$$\frac {\lambda} {2m} \sum_{l=1}^{L}||w^{[l]}||^{2}_F$$

where, $$||w^{[l]}||^{2} _F =\sum^{n[l]} _{i=1} \sum^{n[l-1]} _{j=1} (w^{[l]} _{ij})^2, \quad w: (n[l], n[l-1])$$

We also add lambda value to derivatives in a neural network

$W^{[l]} = W^{[l]} - \alpha \times dW^{[l]}$

$\qquad = W^{[l]} - \alpha \times (\text{(from backprop)} + \frac {\lambda} {m} W^{[l]})$

$\qquad = W^{[l]} - \alpha \frac {\lambda} {m} W^{[l]} - \alpha \times \text{(from backprop)}$

$\qquad = (1 -  \frac {\alpha\lambda} {m})W^{[l]} - \alpha \times \text{(from backprop)}$

where, $dW^{[l]} = \text{(from backprop)} + \frac {\lambda} {m} W^{[l]}$

L2 regularization is also called weight decay since it multiplies weights by $(1 -  \frac {\alpha\lambda} {m})$ value which is less than 1

#### keywords
- regularization parameter $\lambda$
- L1, L2 regularization
- Forbenius norm
- weight decay

#### keypoints
- When using L1 regularization, then $w$ will end up being sparse, which mean $w$ vector will have a lot of zeros in it.
- Having a lot of zeros in a parameter means you need less memory to store the model
- Adding regularization term to the cost function, penalize the weight matrices from being too large
- The intuition of completely zeroing out a bunch of hidden units, or at least reducing the impact of a lot of hidden units, which ends up with what might feel like a simpler neural network.
- If lambda is large, the parameter $w$ is small and $z$ is relatively small and so the activation function if it's tanh, will be relatively linear.

#### questions
- why do we regluarize only $w$ and not $b$?
	- $w$ is a high dimensional parameter vector with a high variance problem whereas $b$ is just a single number so almost all the parameters are in $w$ rather than $b$
- why does regularization reduce overfitting?/ why does it help variance problem?
- how does regularization prevent overfitting?

<a id="2-2"></a>
### 2-2. Dropout Regularization

**Inverted dropout**

- illustrate with layer l=3, keep-prop=0.8

`d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep-prop`

this line will set d3 matrix that has the same dimension as a3 either 1 or 0 according to the randomly generated value compared to the keep-prop value.

- zeroing out

`a3 = np.multiply(a3, d3)`

We multiply a3 by d3 which will result in removing some nodes from the layer, where d3 values are 0.  

- inverted dropout technique

`a3 /= keep-prop`

by dividing the value of keep-prop, it ensures that the expected value of a3 remains the same as well as it makes test time easier because you have less of a scaling problem.

#### keywords
- inverted dropout
- expected value
- diminished network
- spreading out weights
- effect of shrinking the squared norm of the weights

#### keypoints
- zeroing out different hidden units during the iteration
- randomly knocking out units in the network

#### questions
- what does it mean to zero out different hidden units?
- when to use dropout regularization?

<a id="2-3"></a>
### 2-3. Other Regularization methods
#### keywords
- data augmentation
- early stopping

#### questions
- what is the downside of early stopping?

<a id="3"></a>
## 3. Setting up your Optimization Problem
<a id="3-1"></a>
### 3-1. Normalizing inputs

Normalize mean to 0 and variance to 1.

- Subtract mean

$$\mu = \frac {1} {m} \sum^{m} _{i=0} x^{(i)}$$

$$x = x - \mu$$

move the training set until it has zero mean

- Normalize variance

$$\sigma^2 = \frac {1} {m} \sum^{m} _{i=0} x^{(i)} ** 2$$

$$x = x / \sigma$$


#### questions
- why do we normalize input features
	- to avoid having an elongated cost function which makes it harder to find the global minimum during gradient descent
<a id="3-2"></a>
### 3-2. Weight initialization for deep networks

We multiply random initialization by an extra term to adjust the weight's variance in order to avoid vanishing or exploding gradients

for example, one reasonable thing to do would be to set the variance of $W$ to be equal to 1 over $n$
$$Var(w) = \frac {1} {n}$$

where $n$ is the size of the input


- Other variants

$W^{[l]} = np.random.randn(n^{[l]}, n^{[l-1]}) * np.sqrt(\frac {2} {n^{[l-1]}} )$

|Initialization|Term|Activation used|
|--|--|--|
|Xavier|$\sqrt {\frac {1} {n^{[l-1]}}}$|tanh|
|He|$\sqrt {\frac {2} {n^{[l-1]}}}$|ReLU|
|$-$|$\sqrt {\frac {1} {n^{[l-1]} + n^{[l]} } }$|$-$|
