## week4 - Deep Neural Networks
### Summary
> Analyze the key computations underlying deep learning, then use them to build and train deep neural networks for computer vision tasks.
> - Learning Objectives
>    - Describe the successive block sturcture of a deep neural network
>    - Build a deep L-layer neural network
>    - Analyze matrix and vector dimensions to check neural network implementations
>    - Use a chache to pass imformation from forward to back propagation
>    - Explain the role of hyperparameters in deep learning
>    - Build a 2-layer neural network

### Table of contents
1. [Deep Neural Network](#1)
	- 1-1. [Deep neural network notation](#1-1)
	- 1-2. [General forward propagation equation](#1-2)
	- 1-3. [Matrix dimensions](#1-3)
	- 1-4. [Forward and backward functions](#1-4)

<a id="1"></a>
## 1. Deep Neural Network
<a id="1-1"></a>
### 1-1. Deep neural network notation
<img alt="deep neural net" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/6439776b-3c1c-41a8-b097-fc515f6b03fd" width=1000px height=500px>

L=4 (#layers)
$n^{[0]}=n_x=3$, $n^{[1]}=5$, $n^{[2]}=5$, $n^{[3]}=3$, $n^{[4]}=1$

$n^{[l]}$ = #units of layer $l$
$a^{[l]}$ = activations of layer $l$ = $g^{[l]}(z^{[l]})$

$w^{[l]}$ = wights for $z^{[l]}$
$b^{[l]}$ = biases for $z^{[l]}$

<a id="1-2"></a>
### 1-2. General forward propagation equation
$$Z^{[l]} = w^{[l]}A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

<a id="1-3"></a>
### 1-3. Matrix dimensions
$w^{[l]}: (n^{[l]}, n^{[l-1]})$
$b^{[l]}: (n^{[l]}, 1)$
$dw^{[l]}: (n^{[l]}, n^{[l-1]})$
$db^{[l]}: (n^{[l]}, 1)$

$Z^{[l]}, A^{[l]}: (n^{[l]}, m)$
$dZ^{[l]}, dA^{[l]}: (n^{[l]}, m)$

<a id="1-4"></a>
### 1-4. Forward and backward functions
<img alt="overview of input and output" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/a831671f-656e-4ff5-ba52-d24c1b0e9d7f" width=500px height=500px>

layer $l$: $w^{[l]}, b^{[l]}$

| Forward function | Backward function |   
| :------: | :------------------------: |
|input: $a^{[l-1]}$<br>output: $a^{[l]}$<br>cache: $z^{[l]}$|input: $da^{[l]}$, $z^{[l]}$(cache from forward)<br>output: $da^{[l-1]}$, $dw^{[l]}$, $db^{[l]}$|
|$Z^{[l]} = w^{[l]}A^{[l-1]} + b^{[l]}$<br><br>$A^{[l]} = g^{[l]}(Z^{[l]})$|$dZ^{[l]}=A^{[l]}*g^{[l]'}(Z^{[l]})=W^{[l+1]T}dZ^{[l+1]}*g^{[l]'}(Z^{[l]})$<br><br>$dW^{[l]}=\frac{1}{m}dZ^{[l]}A^{[l-1]T}$<br><br>$db^{[l]}=\frac{1}{m}np.sum(dZ^{[l]}, axis=1, keepdims=True)$<br><br>$dA^{[l-1]}=W^{[l]T}dZ^{[l]}$|

### 1-5. Hyperparameters
**parameters**:  $$w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]}, \dots, w^{[l]}, b^{[l]} \quad l = \text{number of layers}$$

**hyperparameters**:

$\qquad \qquad \text{learning rate} \space (\alpha)$
$\qquad \qquad \text{number of iterations}$
$\qquad \qquad \text{number of hidden layer L}$
$\qquad \qquad \text{number of hidden nuits}$
$\qquad \qquad \text{choice of activation functions}$
$\qquad \qquad \text{momentum term}$
$\qquad \qquad \text{mini batch size}$
$\qquad \qquad \text{various forms of regularization}, \dots etc.$

#### keypoints
- hyperparameters are the parameters 
	- that control the parameters w and b
	- that determine the final value of the parameters w and b
