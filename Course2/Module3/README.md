# Module3 - Hyperparameter Tuning, Batch Normalization and Programming Frameworks

### Summary
> Explore Tenserflow, a deep learning framework that allows you to build neural networks quickly and easily, then train a neural network on a Tensorflow dataset.
> - Learning Objectives
>    - Master the process of hyperparameter tuning
>    - Describe softmax classification for multiple classes
>    - Apply batch normalization to make your neural network more robust
>    - Build a neural network in Tensorflow and train it on a Tensorflow dataset.
>    - Describe the purpose and operation of GradientTape
>    - Use tf.Variable to modify the state of a variable
>    - Apply Tensorflow decorators to speed up code
>    - Explain the difference between a variable and a constant

### Table of contents
1. [Hyperparameter Tuning](#1)
	- 1-1. [Tuning Process](#1-1)
	- 1-2. [Using an Appropriate Scale to Pick Hyperparameters](#1-2)
	- 1-3. [Hyperparameters Tuning in Practice: Pandas VS Caviar](#1-3)
2. [Batch Normalization](#2)
3. [Multiclass Classification](#3)


<a id="1"></a>
## 1. Hyperparameter Tuning
<a id="1-1"></a>
### 1-1. Tuning Process
**Hyperparameter Importance**
1. Learning rate ($\alpha$)
2. momentum ($\beta$), mini-batch size,  number of hidden units
3. number of layers, learning rate decay

**Random Search over Grid Search**
It is challenging to predict which hyperparameters will be the most important for the problem at hand. So, using random values rather than a grid search will be more effective

**Coarse to Fine**
If you identify a promising hyperparameter value from the random search, you can narrow down the search space and sample more densely within this smaller region.

<br>

<a id="1-2"></a>
### 1-2. Using an Appropriate Scale to Pick Hyperparameters
Sample on a log scale instead of a uniform scale to better explore the range of hyperparameters.

For example:

$r = -4 * np.random.rand(shape)$

$\alpha = 10^r$

for $\beta$

$1 - \beta = 10^r$

$\beta = 1 - 10^r$

<br>

<a id="1-3"></a>
### 1-3. Hyperparameters Tuning in Practice: Pandas VS Caviar
|Panda Approach<br/>(Babysitting one model)|Caviar Approach<br/>(Train many models in parallel)|
|--|--|
|- Suitable when you have limited computational resources<br/>- Involve carefully monitoring and tweaking one model to optimize its performance|- Suitable when you have ample computational resources<br/> - Involve training many models in parallel with different hyperparameters to find the optimal combination more quickly|

**Considerations**

- **Computational Resources**: The amount of computational power available will influence whether you can adopt the Panda or Caviar approach
- **Data size and Training time**: The size of your dataset and the time it takes to train a model will also affect your hyperparameter tuning strategy

<a id="2"></a>
## 2. Batch Normalization
Batch normalization is a technique to improve the training of neural networks by normalizing the inputs of each layer.

**Normalization**

$$\mu = \frac {1} {m} \sum_{i=0}^{m} x^{(i)}$$

$$x = x - \mu$$

$$\sigma^2 = \frac {1} {m} \sum_{i=0}^{m} {x^{(i)}}^2$$

$$x = x / \sigma$$

**Batch Normalization Steps**
We normalize $Z$ values in hidden layers.
given some intermediate values $Z$ in a neural network:

$$\mu = \frac {1} {m} \sum_{i=0}^{m} Z^{(i)}$$

$$\sigma^2 = \frac {1} {m} \sum_{i=0}^{m} {(Z^{(i)}- \mu)}^2$$

$$Z_{norm}^{(i)} = \frac {Z^{(i)} - \mu} {\sqrt{\sigma^2 + \epsilon}}$$

every component of z has mean 0 and variance 1

but we don't want every hidden layer to have mean 0 and variance 1 in order to better take advantage of non-linearity of activation functions so we use this equation:
$$\tilde Z^{(i)} = \gamma Z_{norm}^{(i)} + \beta$$

where $\gamma$ and $\beta$ are **learnable parameters** of the model.

**Purpose of $\gamma$ and $\beta$**

-   **$\gamma$**: Controls the standard deviation of $\tilde{Z}$.
-   **$\beta$**: Controls the mean of $\tilde{Z}$.

The parameters γ\gammaγ and β\betaβ allow the network to retain the capacity to represent the data in a flexible manner, even after normalization.

If $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$

then $\tilde Z^{(i)} = Z^{(i)}$

**Bias Parameter $b$**
When using batch normalization, the bias parameter $b$ doesn't have any impact since it'll be subtracted out by the mean subtraction step during the batch norm step, instead, we use $\beta$ in order to decide what the $\tilde Z$ is

so you can eliminate $b$ just by setting it to zero.


<a id="3"></a>
## 3. Multiclass Classification

Multiclass classification is a type of classification task where the goal is to assign an input to one of $C$ classes.

**Softmax Function**
The softmax function is commonly used in the output layer of a neural network designed for multiclass classification. It converts the raw output scores (logits) from the network into probabilities that sum to 1, allowing them to be interpreted as class probabilities.

Given:
$$Z = [z_1, z_2, z_3 \dots z_C]$$ are the raw scores for $C$ classes

$$\frac {e^{z_{i}}} {\sum {e^{z_j}}}$$