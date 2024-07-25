
# week2 -  Optimization Algorithms
### Summary
> Develop your deep learning toolbox by adding more advanced optimizations, random minibatching and learning rate decay scheduling to speed up your model.
> - Learning Objectives
>    - Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSprop and Adam
>    - Use random minibatches to accelerate convergence and improve optimization
>    - Describe the benefits of learning rate decay and apply it to your optimization

### Table of contents
1. [Optimization Algorithms](#1)
	- 1-1. [Mini-batch Gradient Descent](#1-1)
	- 1-2. [Exponentially Weighted Average](#1-2)
	- 1-3. [Gradient Descent with Momentum](#1-3)
	- 1-4. [RMSprop](#1-4)
	- 1-5. [Adam](#1-5)
	- 1-6. [Learning Rate Decay](#1-6)

<a id="1"></a>
## 1. Optimization Algorithms
<a id="1-1"></a>
### 1-1. Mini-batch Gradient Descent

**Batch Size Comparison**
|Type|Stochastic|Mini-batch|Batch|
|--|--|--|--|
|Batch size|1|1< batch size < m|m|
|Feature|noisy, never converges|heads more consistently towards the minimum compared to stochastic|relatively low noise, large step size|
|(Dis)advantage|loses the seep-up from vectorization|can leverage the benefits of vectorization|take too much time for each iteration if the training set is huge|


**How to choose batch size**

- If the training set size is less than 2000, batch gradient descent is fine
- Otherwise, it is advisable to use mini-batch sizes, typically ranging from 64 to 512.
- Choosing a mini-batch size that is a power of 2 is advisable because computer memory is typically laid out in powers of 2.
- Ensure the batch size fits in available CPU/GPU memory

<a id="1-2"></a>
### 1-2. Exponentially Weighted Average
A technique that smooths a sequence of values by giving more weight to recent observations.
The exponentially weighted average is defined as: $$V_t = \beta V_{t-1} + (1 - \beta)\theta_t$$

**Understanding $\beta$**
$\beta$ controls the weighting of past observations, with higher $\beta$ value giving a more weight to older observations.

- When $\beta$ is 0.9, it effectively averages the last 10 elements. This is because $0.9^{10} \approx 0.35 \approx \frac {1} {e}$
- When $\beta$ is 0.98, it effectively averages the last 50 elements. This is because $0.98^{50} \approx \frac {1} {e}$

**Bias Correction** 
To get a better estimate in the early stages, bias correction is applied as follows: $$\frac {V_t} {1 - \beta^t}$$

Bias correction compensates for the fact that $V_t$ is initialized at zero and may initially biased towards smaller values. This correction helps provide a more accurate estimate, especially during the initial iterations of the averaging process.


<a id="1-3"></a>
### 1-3. Gradient Descent with Momentum
When the learning rate is too high, the gradient descent process might oscillate. These oscillations can slow down gradient descent and prevent you from using a much larger learning rate. Specifically, if you were to use a much larger learning rate, you might end up overshooting and causing the algorithm to diverge.

In gradient descent, you want your learning to be slower on the vertical axis to avoid oscillations but faster on the horizontal axis to achieve the convergence.

**Momentum**
The concept of momentum in gradient descent helps in smoothing out the steps and accelerating the convergence.

Momentum Parameter $\beta$
It controls how much of the past gradients are retained. A common choice for $\beta$ is 0.9

1. Initialization
<br/>$V_{dW} = 0, V_{db} = 0, \beta_1 = 0.9$

2. On iteration $t$:
- Compute $dW, db$ on the current mini-batch
- Update the *velocities*
<br/>$\quad V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW$
<br/>$\quad V_{db} = \beta_1 V_{db} + (1 - \beta_1) db$

- Update the parameters
<br/>$\quad W := W - \alpha V_{dW}$
<br/>$\quad b := b - \alpha V_{db}$

**Explanation**
The algorithm computes the moving average of the derivatives $dW$ and $db$ for weights and biases respectively.

This moving average helps to smooth out the steps in gradient descent, reducing oscillations and allowing for a more stable convergence.

<a id="1-4"></a>
### 1-4. RMSprop
RMSprop is an optimization algorithm that helps in adjusting the learning rate by considering the magnitude of recent gradients, which stabilizes the training process.

**Algorithm**
1. Initialization
<br/>$S_{dW} = 0, S_{db} = 0, \beta_2 = 0.999$

2. On iteration $t$:
- Compute $dW, db$ on the current mini-batch
- Update the *squared gradient moving averages*
<br/>$\quad S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW^2$
<br/>$\quad S_{db} = \beta_2 S_{db} + (1 - \beta_2) db^2$

- Update the parameters
<br/>$\quad W := W - \alpha \frac {dW} {\sqrt {S_{dW} + \epsilon}}$
<br/>$\quad b := b - \alpha \frac {db} {\sqrt {S_{db} + \epsilon}}$


In the case of 2D where the vertical axis represents $w$ and the horizontal axis represents $b$,
the values of $dW^2$ and  $S_{dW}$ are relatively small and $db^2$ and  $S_{db}$ are relatively large. This results in slower updates in the vertical dimension, stabilizing training process and allowing a larger learning rate without divergence.

In practice, a small value $\epsilon$ is added to the denominator to avoid division by zero:
$$\frac {dW} {\sqrt {S_{dW} + \epsilon}}, \frac {db} {\sqrt {S_{db} + \epsilon}}$$

<a id="1-5"></a>
#### 1-5. Adam
Adam(Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of both RMSprop and momentum. It adapts the learning rate for each parameter and also includes bias correction to provide more accurate estimates in the early stages.

**Algorithm**
Common choices for $\beta_1 = 0.9$, $\beta_2 = 0.999$, and a small value $\epsilon = 10^{-8}$

1. Initialization
<br/>$V_{dW} = 0, V_{db} = 0, \beta_1 = 0.9$
<br/>$S_{dW} = 0, S_{db} = 0, \beta_2 = 0.999$

3. On iteration $t$:
- Compute $dW, db$ on the current mini-batch
- Update the *velocities* 
<br/>$\quad V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW, \quad V_{db} = \beta_1 V_{db} + (1 - \beta_1) db$

- Update the *squared gradient moving averages*
<br/>$\quad S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW^2, \quad S_{db} = \beta_2 S_{db} + (1 - \beta_2) db^2$

- Correct the biases
<br/>$\quad V_{dW}^{corrected} = \frac {V_{dW}} {1 - \beta_1^t}, \quad V_{db}^{corrected} = \frac {V_{db}} {1 - \beta_1^t}$
<br/>$\quad S_{dW}^{corrected} = \frac {S_{dW}} {1 - \beta_2^t}, \quad  S_{db}^{corrected} = \frac {S_{db}} {1 - \beta_2^t}$

- Update the parameters
<br/>$\quad W := W - \alpha \frac {V_{dW}^{corrected}} {\sqrt {S_{dW}^{corrected} + \epsilon}}$
<br/>$\quad b := b - \alpha \frac {V_{db}^{corrected}} {\sqrt {S_{db} ^{corrected}+ \epsilon}}$

**Explanation**
- **Velocites ($V_{dW}, V_{db}$)**: These terms accumulate the exponentially weighted averages of the past gradients.
- **Squared Gradient Moving Averages ($S_{dW}, S_{db}$)**: These terms accumulate the exponentially weighted averages of the squared past gradients.
- **Bias Correction**: This step corrects the biases in the first few iterations to ensure the estimates are not skewed towards zero.
- **Parameter Update**: The parameters are updated using the corrected values of velocities and squared gradients, with $\epsilon$ added to the denominator for numerical stability.

<a id="1-6"></a>
### 1-6. Learning Rate Decay
When using mini-batch gradient descent, the inherent noise due to mini-batch updates means the algorithm will never fully converge. Initially, a relatively large learning rate accelerates learning. As the learning rate decreases over time, the steps become smaller and more precise, allowing the algorithm to oscillate in a tighter region near the minimum, rather than wandering far away even as training continues.

**Decay Formula**
One common approach to learning rate decay is:
$$\alpha = \frac {\alpha_0} {1 + decayRate \times epochNum}$$

**Example**
Given $\alpha_0$ = 0.2, decay_rate = 1
|Epoch|Learning Rate|
|--|--|
|1|0.1|
|2|0.067|
|3|0.05|
|4|0.04|

**Other Methods for Learning Rate Decay**
1. Exponential Decay
<br/>$\alpha = 0.95^{epochNum} \times \alpha_0$

2. Square Root Decay
<br/>$\alpha = \frac {k} {\sqrt {epochNum}} \times \alpha_0$ 
<br/>or
<br/>$\alpha = \frac {k} {\sqrt {t}} \times \alpha_0$
<br/>where $t$ is the batch size

3. Discrete Staircase Decay
- Manually adjust the learning rate at specified intervals or epochs.

Adjusting the learning rate over time helps in balancing the trade-off between convergence speed and stability.