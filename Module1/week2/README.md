## week2 -  Basics of Neural Network programming

### Table of contents
1. [Logistic Regression as a Neural Network](#logistic-regression-as-a-neural-network)
	- [Binary Classification](#binary-classification)
	- [Logistic Regression](#logistic-regression)
	- [Logistic Regression Cost Function](#logistic-regression-cost-function)

## 1. Logistic Regression as a Neural Network
### Binary Classification

#### keywords:
- feature, fecture vector, input fecture vector
- label, output label
- forward propagation, backward propagation
- logistic regression, binary classification, classifier
- train, predict
- size, dimention

#### keypoints:
- organizing the computation of a neural network: forward and backward propagation steps
- logistic regression as a neural network
- image representation in a computer
- transformation of image matrix value into a input feature vector
- deep learning notations
- the default vector: column vector
- matrix vs vector in numpy representation

#### questions:
- what are some important techniques when implementing a neural network?
- why is it better to process the entire traning set without using a explicity for loop to loop over the entire training set?
- why the computations in learning a neural network can be organized in the forward propagation and a separated back propagation?
- why the convention of stacking data in columns to group examples representing input and output matrices is better than the other way(stacking in rows)?
- what is the number of columns and rows of the input matrix X by doing so? 
	- : $m$, $n_x$


### Logistic Regression
#### keywords:
- logistic regression
- sigmoid

#### keypoints:
- parameter notation for deep learning

#### questions:
- given the input $x$ and parameters $w, b$, how do we generate the output $\hat y$?
- what happends when the $Z$ of sigmoid function becomes very large negative number or very small number?
- why is it better to use the conventional notation for parameters separated in $W$ and $b$ rather than having them as $\theta$ all together in deep learning?
- what are the parameters for logistic regression? 
	- : W, as $n_x$ dimensional vector, and b, as a real number

### Logistic Regression Cost Function
$y^{(i)} = \sigma(w^Tx^{(i)} + b)$, where $\sigma(z) = \frac{1}{1 + e^{-z^{(i)}}}$, $z^{(i)} =w^Tx^{(i)} + b$

Given {$(x^{(1)}, y^{(1)}),\dots,(x^{(m)}, y^{(m)})$}, want $\hat y^{(i)} \approx y^{(i)}$

Loss(error) function: $$L(\hat y, y) = -(y \log(\hat y) + (1-y)\log(1-\hat y))$$

Cost function:$$J(w,b) = \frac{1}{m} \sum^m_{i=0} L(\hat y^{(i)}, y^{(i)} ) = -\frac{1}{m} \sum^m_{i=0}[(y^{(i)} \log(\hat y^{(i)}) + (1-y^{(i)})\log(1-\hat y^{(i)}))]$$

#### keywords:
- convex, non-convex
- ground true label

#### keypoints:
- the choice of loss function for logistic regression compared to the one used for linear regression
- loss function for training example, cost function for parameters of the algorithm

#### questions:
- why we don't use the MSE cost function for logistic regression?
	- : because the optimization problem becomes non convex (multiple local optimums) so gradient descent may not find the global optimum
- what is the difference between loss function and cost function?
	- : the loss function computes the error for a single training example; the cost function is the average of the loss function of the entire training set.
- why do we need/use loss/error function?
	- : to measure how well the algorithm is doing/ how good the output $\hat y$ is when the true label is $y$
- what is the training procedure and what is loss function for?
- what does it mean finding parameters $W$ and $b$ that minimize the overall cost function $J$?
