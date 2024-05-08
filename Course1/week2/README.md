## week2 -  Basics of Neural Network programming
### Summary
> Set up a machine learning problem with a neural network mindset and use vectorization to speed up your models.
> - Learning Objectives
>    - Build a logistic regression model structured as a shallow neural network
>    - Build the general architecture of a learning algorithm, including parameter initialization, cost function and gradient calculation, and optimization implementation (gradient descent)
>    - Implement computationally efficient and highly vectored versions of models
>    - Compute derivatives for logistic regression using a backpropagation mindset
>    - Use Numpy functions and Numpy matrix/vector operations
>    - Implement vectorization across multiple training examples
>    - Explain the concept of broadcasting

### Table of contents
1. [Logistic Regression as a Neural Network](#logistic-regression-as-a-neural-network)
	- [Binary Classification](#binary-classification)
	- [Logistic Regression](#logistic-regression)
	- [Logistic Regression Cost Function](#logistic-regression-cost-function)
	- [Gradient Descent](#gradient-descent)
	- [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
2. [Python and Vectorization](#2)
3. [General Architecture of the learning algorithm](#3)
4. [Building the parts of the algorithm](#4)
	- [Forward and Backward propagation](#4-1)

## 1. Logistic Regression as a Neural Network
### Binary Classification

#### keywords:
- feature, feature vector, input feature vector
- label, output label
- forward propagation, backward propagation
- logistic regression, binary classification, classifier
- train, predict
- size, dimension

#### keypoints:
- organizing the computation of a neural network: forward and backward propagation steps
- logistic regression as a neural network
- image representation in a computer
- transformation of image matrix value into an input feature vector
- deep learning notations
- the default vector: column vector
- matrix vs vector in numpy representation

#### questions:
- what are some important techniques when implementing a neural network?
- why is it better to process the entire training set without using an explicit for loop to loop over the entire training set?
- why the computations in learning a neural network can be organized in forward propagation and a separated backpropagation?
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
- what happens when the $Z$ of the sigmoid function becomes a very large negative number or a very small number?
- why is it better to use the conventional notation for parameters separated in $W$ and $b$ rather than having them as $\theta$ together in deep learning?
- what are the parameters for logistic regression? 
	- : W, as $n_x$ dimensional vector, and b, as a real number

### Logistic Regression Cost Function
$y^{(i)} = \sigma(w^Tx^{(i)} + b)$, where $\sigma(z) = \frac{1}{1 + e^{-z^{(i)}}}$, $z^{(i)} =w^Tx^{(i)} + b$

Given { $(x^{(1)}, y^{(1)}),\dots,(x^{(m)}, y^{(m)})$ }, want $\hat y^{(i)} \approx y^{(i)}$

Loss(error) function: $$L(\hat y, y) = -(y \log(\hat y) + (1-y)\log(1-\hat y))$$

Cost function: $$J(w,b) = \frac{1}{m} \sum^m_{i=0} L(\hat y^{(i)}, y^{(i)} ) = -\frac{1}{m} \sum^m_{i=0}[(y^{(i)} \log(\hat y^{(i)}) + (1-y^{(i)})\log(1-\hat y^{(i)}))]$$

#### keywords:
- convex, non-convex
- ground true label

#### keypoints:
- the choice of loss function for logistic regression compared to the one used for linear regression
- loss function for training example, cost function for parameters of the algorithm

#### questions:
- why we don't use the MSE cost function for logistic regression?
	- : because the optimization problem becomes non-convex (multiple local optimums) so gradient descent may not find the global optimum
- what is the difference between loss function and cost function?
	- : the loss function computes the error for a single training example; the cost function is the average of the loss function of the entire training set.
- why do we need/use the loss/error function?
	- : to measure how well the algorithm is doing/ how good the output $\hat y$ is when the true label is $y$
- why do we need cost function?
 	- : to change the parameters $w$ and $b$
- what is the training procedure and what is loss function for?
- what does it mean finding parameters $W$ and $b$ that minimize the overall cost function $J$?

### Gradient Descent
repeat: $$w := w - \alpha \frac{\partial J(w, b)}{\partial w} $$
 $$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$$
 $\alpha$ : learning rate
  $\frac{df(w)}{dw}$ $\approx dw$ : derivative
#### keywords:
- learning rate
- gradient descent
- local optima, global optima
- convergence

#### keypoints:
- initial values for parameters $W, b$ (initial step at the convex function to descent)
- iterations of gradient descent

#### questions:
- why we don't usually initialize parameters randomly for logistic regression?
- what is the role of learning rate ($\alpha$)?
	- : it controls how big a step we take on each iteration during gradient descent
- what is the definition of a derivative?
	- : a slop of a function at a particular point
- does a convex function always have multiple local optima?
	- : false

### Logistic Regression Gradient Descent
$z =w^Tx + b$

$\hat y = a = \sigma(z) = \frac{1}{1 + e^{-z}}$

$L(a, y) = -(y \log(a) + (1-y)\log(1-a))$

<div>
<br><br>
<img src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/2e67b6bc-2bcb-4af2-8060-e5aa389ad59e" alt="gradient descent example1">
  <figcaption>Computation graph: two input features x1, x2</figcaption>
<br><br>
</div>
derivatives:

$\frac{\partial L(a, y)}{\partial a} \approx da = -\frac{y}{a} + \frac{1-y}{1-a}$

$\frac{\partial a}{\partial z} = a(1-a)$

$\frac{\partial L}{\partial z} \approx dz = \frac{\partial L}{\partial a}  \frac{\partial a}{\partial z} = a-y$

$\frac{\partial L}{\partial w_1} \approx dw_1 = x_1  \frac{\partial L}{\partial z} \approx x_1  dz$

$\frac{\partial L}{\partial w_2} \approx dw_2 = x_2  \frac{\partial L}{\partial z} \approx x_2  dz$

$\frac{\partial L}{\partial b} \approx db = \frac{\partial L}{\partial z} \approx dz$

#### keywords:
- forward functions, backward functions
- chain rule

#### keypoints:
- forward propagation/pass: to compute the output of the neural network
- left-to-right pass: to compute the value of $J$
- backward propagation/pass: to compute the gradients, derivatives
- right-to-left pass: to compute the derivatives
- we want to optimize/minimize the cost function $J$
- the way of computing derivatives

#### questions:
- what does it mean to optimize/minimize the cost function?
- one step of backward propagation on a computation graph yields derivatives of the finial output variable
	- : true
- what does it mean to propagate the change of a parameter to the output value?
- what is the chain rule when calculating derivatives?

<a id='2'></a>
## 2. Python and Vectorization

- Iterative version
```python
J = 0, dw1 = 0, dw2 = 0, db = 0
for i = 1 to m:
	z[i] = w.T x[i] + b
	a[i] = σ(z[i])
	J += -[y[i]log(a[i]) + (1 - y[i])log(1 - a[i])]
	dz[i] = a[i] - y[i]
	dw1 += x1[i]dz[i]
	dw2 += x2[i]dz[i]
	db += dz[i]
J = J/m, dw1 = dw1/m, dw2 = dw2/m, db = db/m
```
- Vectorized version

$Z = w^TX + b = np.dot(w.T, X) + b$

$A = \sigma(Z)$

$dZ = A - Y$

$dw = \frac{1}{m}XdZ^T$

$db = \frac{1}{m}np.sum(dZ)$

$$w := w - \alpha dw$$ 

$$b := b - \alpha db$$


#### keywords:
- vectorization
- broadcasting
- normalization

#### keypoints:
- a single iteration of gradient descent
- normalization: makes gradient descent converge faster
- sigmoid function and its gradient
- keep vector/matrix dimensions straight
- numpy built-in functions

#### questions:
- what is python broadcasting
- what are dot/outer/elementwise products
- what is vectorization


<a id='3'></a>
## 3. General Architecture of the learning algorithm

It's time to design a simple algorithm to distinguish cat images from non-cat images.

You will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**

<img width="518" alt="LogReg_kiank" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/bf22b2c2-53c4-47aa-84c9-237c92722ec0" style="width:650px;height:400px;">

**Mathematical expression of the algorithm**:

For one example $x^{(i)}$: 
$$z^{(i)} = w^T x^{(i)} + b$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$ 
$$\mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})$$

The cost is then computed by summing over all training examples:
 $$J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$$

**Key steps**:
In this exercise, you will carry out the following steps: 
   - Initialize the parameters of the model
   - Learn the parameters for the model by minimizing the cost  
   - Use the learned parameters to make predictions (on the test set)
   - Analyse the results and conclude

<a id='4'></a>
## 4. Building the parts of the algorithm

The main steps for building a Neural Network are:

1.  Define the model structure (such as the number of input features)
2.  Initialize the model's parameters
3.  Loop:
    -   Calculate current loss (forward propagation)
    -   Calculate current gradient (backward propagation)
    -   Update parameters (gradient descent)

<a id='4-1'></a>
### Forward and Backward propagation

"forward" and "backward" propagation steps for learning the parameters that compute the cost function and its gradient.


Forward Propagation:
- You get X
- You compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
- You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$

Here are the two formulas you will be using: 

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTgzNzYwNDg2OF19
-->
