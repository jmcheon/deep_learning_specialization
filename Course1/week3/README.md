## week3 - One hidden layer Neural Networks
### Summary
> Build a neural network with one hidden layer using forward propagation and backpropagation.
> - Learning Objectives
>    - Describe hidden units and hidden layers
>    - Use units with a non-linear activation function, such as tanh
>    - Implement forward and backward propagation
>    - Apply random initialization to your neural network
>    - Increase fluency in Deep Learning notations and Neural Network Representations
>    - Implement a 2-class classification neural network with a single hidden layer
>    - Compute the cross entropy loss

### Table of contents
1. [Shallow Neural Network](#1)
	- [Neural Network Representiation](#1-1)

<a id="1"></a>
## Shallow Neural Network
<a id="1-1"></a>
### Neural Network Representiation
$a^{[l]}_i$, where $l$ is the $l^{th}$ layer and $i$ is the $i^{th}$ node

$x = a^{[0]}$

$\hat y = a^{[\text{The number of layers}]}$
<img width="518" alt="neural network representation" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/e9b5636d-7aab-490e-b6d3-b67c68b119f7" style="width:650px;height:400px;">

$z^{[1]}_1 = w^{[1]T}_1x + b^{[1]}_1, a^{[1]}_1 =\sigma(z^{[1]}_1)$

$z^{[1]}_2 = w^{[1]T}_2x + b^{[1]}_2, a^{[1]}_2 =\sigma(z^{[1]}_2)$

$z^{[1]}_3 = w^{[1]T}_3x + b^{[1]}_3, a^{[1]}_3 =\sigma(z^{[1]}_3)$

$z^{[1]}_4 = w^{[1]T}_4x + b^{[1]}_4, a^{[1]}_4 =\sigma(z^{[1]}_4)$

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
- we don't count the input layer as a official layer when we count layers of a neural network
- input layer = layer zero
- hidden layers and output layers have paramaters $w$ and $b$ associated with them
- parameter dimensions 
