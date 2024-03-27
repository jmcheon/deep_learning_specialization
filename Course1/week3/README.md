
## week3 - One Hidden Layer Neural Networks
### Summary
> Build a neural network with one hidden layer using forward propagation and backpropagation.
> - Learning Objectives
>    - Describe hidden units and hidden layers
>    - Use units with a non-linear activation function, such as Tanh
>    - Implement forward and backward propagation
>    - Apply random initialization to your neural network
>    - Increase fluency in Deep Learning notations and Neural Network Representations
>    - Implement a 2-class classification neural network with a single hidden layer
>    - Compute the cross entropy loss

### Table of contents
1. [Shallow Neural Network](#1)
	- 1-1. [Neural Network Representiation](#1-1)
	- 1-2. [Vectorizing Across Multiple Examples](#1-2)
	- 1-3. [Activation Functions](#1-3)
	- 1-4. [Gradient Descent for Neural Networks](#1-4)

<a id="1"></a>
## 1. Shallow Neural Network

<a id="1-1"></a>
### 1-1. Neural Network Representation

| Perceptron | 2 Layers Neural Network |   
| :------: | :------------------------: |
|<img width="518" alt="perceptron" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/fd70daa4-a69a-461c-b764-d089c6f02dcd" width=500px height=200px>|<img width="518" alt="neural network representation" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/e9b5636d-7aab-490e-b6d3-b67c68b119f7" width=500px height=200px>|
|$z = W^Tx +b$<br>$a=\sigma {(z)}$|$a^{[l]}_i$, where $l$ is the $l^{th}$ layer and $i$ is the $i^{th}$ node<br>$x = a^{[0]}, \space \hat y = a^{[\text{The number of layers}]}$<br><br>$z^{[1]}_1 = w^{[1]T}_1x + b^{[1]}_1, \space a^{[1]}_1 =\sigma(z^{[1]}_1)$<br>$z^{[1]}_2 = w^{[1]T}_2x + b^{[1]}_2, \space a^{[1]}_2 =\sigma(z^{[1]}_2)$<br>$z^{[1]}_3 = w^{[1]T}_3x + b^{[1]}_3,\space  a^{[1]}_3 =\sigma(z^{[1]}_3)$<br>$z^{[1]}_4 = w^{[1]T}_4x + b^{[1]}_4, \space a^{[1]}_4 =\sigma(z^{[1]}_4)$|


##### 2 Layers Neural Network

$$z^{[1]}= \underbrace{\overbrace{\begin{bmatrix}
\text{-----}  \space w^{[1]T}_1 \text{-----} \space\\
\text{-----}  \space w^{[1]T}_2 \text{-----}  \space \\
\text{-----}  \space w^{[1]T}_3 \text{-----}  \space \\
\text{-----}  \space w^{[1]T}_4 \text{-----}  \space
\end{bmatrix}
}^{W^{[1]}}} _{(4, 3)}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\end{bmatrix}
+
\begin{bmatrix}
b^{[1]}_1 \\
b^{[1]}_2 \\
b^{[1]}_3 \\
b^{[1]}_4 \\
\end{bmatrix} =
\begin{bmatrix}
w^{[1]T}_1 x + b^{[1]}_1 \\
w^{[1]T}_2 x + b^{[1]}_3 \\
w^{[1]T}_3 x + b^{[1]}_3 \\
w^{[1]T}_4 x + b^{[1]}_4
\end{bmatrix} =
\begin{bmatrix}
z^{[1]}_1 \\
z^{[1]}_2 \\
z^{[1]}_3 \\
z^{[1]}_4 \\
\end{bmatrix}$$

$$a^{[1]}=
\begin{bmatrix}
a^{[1]}_1 \\
a^{[1]}_2 \\
a^{[1]}_3 \\
a^{[1]}_4 \\
\end{bmatrix} = 
\sigma (z^{[1]})$$

$$\underbrace{z^{[1]}} _{(4, 1)} = \underbrace{W^{[1]}} _{(4, 3)}\underbrace{x} _{(3, 1)} + \underbrace{b^{[1]}} _{(4, 1)} = \underbrace{W^{[1]}} _{(4, 3)}\underbrace{a^{[0]}} _{(3, 1)} + \underbrace{b^{[1]}} _{(4, 1)}$$

$$\underbrace{a^{[1]}} _{(4, 1)} =\sigma(\underbrace{z^{[1]}} _{(4, 1)})$$

$$\underbrace{z^{[2]}} _{(1, 1)}  = \underbrace{W^{[2]}} _{(1, 4)}\underbrace{a^{[1]}} _{(4, 1)} + \underbrace{b^{[2]}} _{(1, 1)}$$

$$\underbrace{a^{[2]}} _{(1, 1)} =\sigma(\underbrace{z^{[2]}} _{(1, 1)})$$

#### keywords
- neurons/units/nodes
- layers(input, hidden, output layers)

#### keypoints
- we don't count the input layer as an official layer when we count layers of a neural network
- input layer = layer zero
- hidden layers and output layers have parameters $w$ and $b$ associated with them
- parameter dimensions 
- we stack different nodes in a layer vertically to form a corresponding vector ($z, a$)

<a id="1-2"></a>
### 1-2. Vectorizing Across Multiple Examples

<img alt="nerual net vectorization" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/33602743-c9ba-43dc-bb17-946086f4cd0e" width=2000px height=500px>

![]()
##### Iterative version for $m$ examples
for i = 1 to m: 

$\qquad z^{[1] (i)} = W^{[1]}x^{(i)} + b^{[1]}$ 

$\qquad a^{[1] (i)} =\sigma(z^{[1] (i)})$ 

$\qquad z^{[2] (i)} = W^{[2]}a^{[1] (i)} + b^{[2]}$ 

$\qquad a^{[2] (i)} =\sigma(z^{[2] (i)})$ 

##### Vectorized version for $m$ examples
$Z^{[1]} = W^{[1]}X + b^{[1]} = W^{[1]}A^{[0]}  + b^{[1]}$

$A^{[1]} =\sigma(Z^{[1]})$

$Z^{[2]} = W^{[2]}A^{[1]} + b^{[1]}$

$A^{[2]} =\sigma(Z^{[2]})$


$$X=  \overbrace{\begin{bmatrix}
x^{(1)}_1 & x^{(2)}_1 & \dots & x^{(m)}_1 \\
x^{(1)}_2 & x^{(2)}_2 & \dots & x^{(m)}_2 \\
x^{(1)}_3 & x^{(2)}_3 & \dots & x^{(m)}_3 \\
\end{bmatrix}}^{\xleftrightarrow{\textbf {training examples}}}
\updownarrow \textbf{features}$$
<br>

$$Z^{[1]}=\overbrace{\begin{bmatrix}
z^{[1] (1)}_1 & z^{[1] (2)}_1 & \dots & z^{[1] (m)}_1 \\
z^{[1] (1)}_2 & z^{[1] (2)}_2 & \dots & z^{[1] (m)}_2 \\
z^{[1] (1)}_3 & z^{[1] (2)}_3 & \dots & z^{[1] (m)}_3 \\
z^{[1] (1)}_4 & z^{[1] (2)}_4 & \dots & z^{[1] (m)}_4 \\
\end{bmatrix}}^{\xleftrightarrow{\textbf {training examples}}}
\updownarrow \textbf{hidden units of \textit{layer 1}}, \space Z^{[2]}=\overbrace{\begin{bmatrix}
z^{[2] (1)} & z^{[2] (2)} & \dots & z^{[2] (m)} \\
\end{bmatrix}}^{\xleftrightarrow{\textbf {training examples}}}
\updownarrow \textbf{hidden units of \textit{layer 2}}$$
<br>

$$A^{[1]}=\overbrace{\begin{bmatrix}
a^{[1] (1)}_1 & a^{[1] (2)}_1 & \dots & a^{[1] (m)}_1 \\
a^{[1] (1)}_2 & a^{[1] (2)}_2 & \dots & a^{[1] (m)}_2 \\
a^{[1] (1)}_3 & a^{[1] (2)}_3 & \dots & a^{[1] (m)}_3 \\
a^{[1] (1)}_4 & a^{[1] (2)}_4 & \dots & a^{[1] (m)}_4 \\
\end{bmatrix}}^{\xleftrightarrow{\textbf {training examples}}}
\updownarrow \textbf{hidden units of \textit{layer 1}}, \space A^{[2]}=\overbrace{\begin{bmatrix}
a^{[2] (1)} & a^{[2] (2)} & \dots & a^{[2] (m)} \\
\end{bmatrix}}^{\xleftrightarrow{\textbf {training examples}}}
\updownarrow \textbf{hidden units of \textit{layer 2}}$$


<a id="1-3"></a>
### 1-3. Activation Functions

#### keywords
- activation function($g$) - either nonlinear or linear(identity)
- tanh(mathematically a shifted version of sigmoid function)
- rectified linear nuit(ReLU), leacky ReLU

#### keypoints
- choice of activation functions(optimization)
- we no longer use sigmoid as activation function for hidden layers except for the output layer(binary classification)
- the downsides of both sigmoid and tanh
- the purpose of activation functions

#### questions
- why do we use different activation functions for hidden layers and output layers?
- why is it better to use tanh than sigmoid for hidden layers?
- what does it mean to slow down gradient descent?
- what are the advantages of using either ReLU or leaky ReLU?
- why do we use activation functions and what happens when we don't have activation functions in neural networks(or have only linear/identity activation functions)?

<a id="1-4"></a>
### 1-4. Gradient Descent for Neural Networks
**Parameters:** $\underbrace{w^{[1]}} _{(n^{[1]}, n^{[0]})}, \underbrace{b^{[1]}} _{(n^{[1]}, 1)}, \underbrace{w^{[2]}} _{(n^{[2]}, n^{[1]})}, \underbrace{b^{[2]}} _{(n^{[2]}, 1)}\qquad n^{[0]} = n_x, \space n^{[1]}, \space n^{[2]} =1$

**Cost function:** $$J(w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]})=\frac{1}{m} \sum^{n}_{i=0}L(\underbrace{\hat y} _{a^{[2]}}, y)$$

**Gradient descent:** 

$\qquad Repeat \space$ {

$$\text{compute predictions}\space (\hat y^{(i)}, \space i = 1 \dots m)$$

$$dw^{[1]} = \frac{\partial J}{\partial w^{[1]}}, \space db^{[1]} = \frac{\partial J}{\partial b^{[1]}}$$

$$w^{[1]} := w^{[1]} - \alpha \space dw^{[1]}$$

$$b^{[1]} := b^{[1]} - \alpha \space db^{[1]}$$

$$dw^{[2]} = \frac{\partial J}{\partial w^{[2]}}, \space db^{[2]} = \frac{\partial J}{\partial b^{[2]}}$$

$$w^{[2]} := w^{[2]} - \alpha \space dw^{[2]}$$

$$b^{[2]} := b^{[2]} - \alpha \space db^{[2]}$$

$\qquad$ }

#### Formulas for computing derivatives
| Forward Propagation | Backward Propagation |   
| :------: | :------------------------: |
|$Z^{[1]} = W^{[1]}A^{[0]}  + b^{[1]}$<br>$A^{[1]} =g(Z^{[1]})$<br>$Z^{[2]} = W^{[2]}A^{[1]} + b^{[1]}$<br>$A^{[2]} =g(Z^{[2]})$|$dZ^{[2]}=A^{[2]}-Y, \space Y=[y^{(1)}, y^{(1)}, \dots, y^{(m)}]$<br>$dW^{[2]}=\frac{1}{m}dZ^{[2]}A^{[1]T}$<br>$db^{[2]}=\frac{1}{m}np.sum(dZ^{[2]}, axis=1, keepdims=True)$<br>$dZ^{[1]}=\underbrace{W^{[2]T}dZ^{[2]}} _{(n^{[1]}, m)}\space * \space \underbrace{g^{[1]'}(Z^{[1]})} _{(n^{[1]}, m)}$<br>$dW^{[1]}=\frac{1}{m}dZ^{[1]}X^{T}$<br>$db^{[1]}=\frac{1}{m}np.sum(dZ^{[1]}, axis=1, keepdims=True)$|

#### keypoints
- how to compute the partial derivative terms for parameters

#### questions
- why is it important to initialize parameters randomly rather than to all zeros?
- why does initializing weights to all zeros not work when applying gradient descent?
