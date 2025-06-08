## Module1 - Recurrent Neural Networks

### Table of contents
1. [Recurrent Neural network](#1)
	- [Notation](#1-1)
	- [Forward propagation](#1-2)
	- [Backpropagation through time](#1-3)
	- [Different types of RNN](#1-4)

<a id="1"></a>
### 1. Recurrent Neural network
<img width="779" alt="RNN" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/8686205f-2633-4930-bd25-e04d06930718">

<a id="1-1"></a>
#### Notation

different examples in the training set can have different length  $T_x^{(i)}$, $T_y^{(i)}$
$W_{ax}$: the second index means that this $W_{ax}$ is going to be multiplied by some X-like quantity 

##### keypoints
- the parameters it uses for each time step are shared

why not a standard network?
- inputs and outputs can be different lengths in different examples
- doesn't share features learned across different positions of text

<a id="1-2"></a>
#### Forward propagation

<img width="961" alt="rnn_step_forward" src="https://github.com/jmcheon/deep_learning_specialization/assets/40683323/ebfdf078-de53-4be0-83a1-893a65abedc9">


$$a^{<t>}=g_1(\underbrace{W_{aa}} _{(100, 100)} \underbrace{a^{< t - 1 >}} _{100} + \underbrace{W _{ax}} _{(100, 10000)} \underbrace{x^{< t >}} _{10000} + b_a)$$

$$\hat y^{< t >}=g_2(W _{ya}a^{< t >} + b_y)$$


simplified RNN notation

$$a^{< t >}=g_1(W_{a}[a^{< t - 1 >},x^{< t >}] + b_a)$$

$$\underbrace{W_{a}} _{(100, 10100)} = [\underbrace{W _{aa}} _{100}  \space \underbrace{W _{ax}} _{10000}]$$ 

$$[\underbrace{a^{< t - 1 >}} _{100}, \underbrace{x^{< t >}} _{10000}] =
\begin{bmatrix}
a^{< t - 1 >}\\
x^{< t >} 
\end{bmatrix} \updownarrow{10100}$$


$$\hat y^{< t >}=g_2(W_{y}a^{< t >} + b_y)$$

<a id="1-3"></a>
#### Backpropagation through time

$$L^{< t >}(\hat y^{< t >}, y^{< t >})=-y^{< t >}\log(y^{< t >}) - (1 - y^{< t >})\log(1 - y^{< t >})$$

$$L(\hat y, y) = \sum_{t=1}^{T_y} L^{< t >}(\hat y^{< t >}, y^{< t >})$$

<a id="1-4"></a>
#### Different types of RNN
| One to One | One to Many | Many to One |
| :------: | :------------------------: |:------------------------: |
|<img alt="one to one" src="https://github.com/jmcheon/GPTs/assets/40683323/17d64649-89a8-446f-9bfd-4b988071dc5b" width=500px height=200px>|<img alt="one to many" src="https://github.com/jmcheon/GPTs/assets/40683323/b427297d-1f27-41d6-aa89-d44430c6eb2a" width=500px height=200px>|<img alt="many to one" src="https://github.com/jmcheon/GPTs/assets/40683323/c6974017-10e5-4dbc-a399-9619e0737835" width=500px height=200px>|
||music generation|sentiment classification|

| Many to many | Many to one + one to many | 
| :------: | :------------------------: |
|<img alt="many to many" src="https://github.com/jmcheon/GPTs/assets/40683323/7632becc-85a2-4ac5-92a4-3859a443ef7f" width=500px height=200px>|<img alt="sec2sec" src="https://github.com/jmcheon/GPTs/assets/40683323/d6f96784-1cb5-4678-959c-073658525a78" width=500px height=200px>|
|name entity recognition|machine translation|

