# week1 - ML Strategy
### Summary
> Streamline and optimize your ML production workflow by implementing strategic guidelines for goal-setting and applying human-level performance to help define key priorities.
> - Learning Objectives
>    - Explain why Machine Learning strategy is important
>    - Apply satisficing and optimizing metrics to set up your goal for ML projects
>    - Choose a correct train/dev/test split of your dataset
>    - Define human-level performance
>    - Use human-level performance to define key priorities in ML projects
>    - Make the correct strategic decision based on observations of performances and the dataset

### Table of contents
1. [Introduction to ML Strategy](#1)
	- 1-1. [Why ML Strategy](#1-1)
	- 1-2. [Orthogonalization](#1-2)
2. [Setting up your Goal](#2)
	- 2-1. [Single Number evaluation Metric](#2-1)
	- 2-2. [Satisficing and Optimizing Metric](#2-2)
	- 2-3. [Train/Dev/Test Distributions](#2-3)
	- 2-4. [Size of the Dev and Test Sets](#2-4)
	- 2-5. [When to Change Dev/Test Sets and Metrics](#2-5)
3. [Comparing to Human-level Performance](#3)
	- 3-1. [Why Human-level Performance?](#3-1)
	- 3-2. [Avoidable Bias](#3-2)
	- 3-3. [Understanding Human-level Performance](#3-3)
	- 3-4. [Surpassing Human-level Performance](#3-4)
	- 3-5. [Improving your Model Performance](#3-5)

<a id="1"></a>
## 1. Introduction to ML Strategy
<a id="1-1"></a>
### 1-1. Why ML Strategy
When you're building an AI model and you want to improve its performance, there are numerous strategies and methods to consider. Here, you can learn various strategies and ways of analyzing a machine learning problem that can help you identify the most promising approaches to try.

**Ideas**:
- Collect more data
- Collect more diverse training set
- Train algorithm longer with gradient descent
- Try Adam instead of gradient descent
- Try a bigger network
- Try smaller network
- Try dropout
- Add $L_2$ regularization
- Network architecture
	- Activation functions
	- Number of hidden units

<a id="1-2"></a>
### 1-2. Orthogonalization
Orthogonalization refers to the strategy of adjusting one aspect of the model at a time instead of modifying multiple factors simultaneously. This helps in isolating the impact of each change and understanding which adjustments are most effective.

**Chain of assumptions in ML**
1. Fit training set well on cost function ($\approx$ human-level performance)
	- use a bigger network
	- use Adam optimizer
2. Fit dev set well on cost function
	- apply regularization
	- increase the size of training set
3. Fit test set well on cost function
	- use a larger dev set
4. Performs well in real world
	- change dev set or cost function as necessary

<br>

<a id="2"></a>
## 2. Setting up your Goal
<a id="2-1"></a>
### 2-1. Single Number evaluation Metric

**Precision**: of examples recognized as true, what percentage are actually true

**Recall**: what percentage of actual true examples are recognized as true

**F1 score**: harmonic mean(average) of precision and recall

$$\frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$


Setting up dev set, which is you're measuring precision and recall and the evaluation metric defines the target you want to aim at.

**Dev set + single number evaluation metric => speed up iterative process of improving ML algorithm**

<br>

<a id="2-2"></a>
### 2-2. Satisficing and Optimizing Metric

Example:
<table>
	<tr>
		<th>Classifier</th>
		<th>Accuracy</th>
		<th>Running Time</th>
	</tr>
	<tr>
		<td>A</td>
		<td>90%</td>
		<td>80ms</td>
	</tr>
	<tr>
		<td>B</td>
		<td>92%</td>
		<td>95ms</td>
	</tr>
	<tr>
		<td>C</td>
		<td>95%</td>
		<td>1500ms</td>
	</tr>
</table>

cost = accuracy - 0.5 * running time


Objective: maximize accuracy subject to running time <= 100 ms

- **Optimizing metric:** Accuracy (maximize as well as possible)
- **Satisficing metric:** Running time (must be good enough)

When there are $n$ number of metrics, it is reasonable to choose one metric as the **optimizing metric** and $n - 1$ number of **satisficing metrics**.

<br>

<a id="2-3"></a>
### 2-3. Train/Dev/Test Distributions
Guideline:

- Choose the dev set and test set to reflect the data you expect to get in the future and consider important to do well on.
- Dev and Test sets should come from the same distribution.
- If the data comes from different distributions, randomly shuffle data into dev and test set.

The data used should resemble the future data you expect to encounter and want to perform well on.

<br>

<a id="2-4"></a>
### 2-4. Size of the Dev Set and Test Set
**Size of Test Set**

- Ensure test set is large enough to give high confidence in the overall performance of your system.

**Purpose of Dev Set**

- It is to tune and help evaluate different ideas and to pick the better algorithm.

**Purpose of Test Set**

- It is to evaluate how good the final system is after development.

<br>

<a id="2-5"></a>
### 2-5. When to change Dev/Test Sets and Metrics

If the evaluation metric does not give a correct rank order preference for better algorithms, it is time to redefine the evaluation metric.

**Orthogonalization for a Classifier**

It is better to think separately the two steps

1. Define a metric (Set the target)
2. Determine how to perform well on that metric (Aim at the target)

If a better algorithm negatively impacts undesired data, adjust the cost function to include a penalty weight to mitigate this behavior.

If real-world data doesn't correspond to the dev/test sets, consider changing dev/test sets to reflect real-world data.

<br>

<a id="3"></a>
## 3. Comparing to Human-level Performance
<a id="3-1"></a>
### 3-1. Why Human-level Performance?

There are main two reasons to compare ML systems to human-level performance:

1. **Feasibility**: ML algorithms can often match and exceed human-level performance in many applications, making them highly competitive and practical.
2. **Efficiency**: The workflow for designing and building an ML system is more efficient when the task is something that humans can also do well.

Progress tends to be relatively rapid as you approach the human-level performance but slows down once surpassing it. The theoretical limit of performance is known as Bayes Optimal Error which is the best possible error.

$$\text{Bayes Optimal Error = Bayesian Optimal Error = Bayes Error}$$

**Why compare to human-level performance?**
Humans are quite good at a lot of tasks. When ML performs worse than humans, you can:
- Get labeled data from humans
- Gain insights from manual error analysis: understand why humans got it right
- Better analysis of bias/variance issues

<br>


<a id="3-2"></a>
### 3-2. Avoidable Bias
Example:

<table>
	<tr>
		<th></th>
		<th>Scenario 1</th>
		<th>Scenario 2</th>
	</tr>
	<tr>
		<td>Human Error</td>
		<td>1%</td>
		<td>7.5%</td>
	</tr>
	<tr>
		<td>Training Error</td>
		<td>8%</td>
		<td>8%</td>
	</tr>
	<tr>
		<td>Dev Error</td>
		<td>10%</td>
		<td>10%</td>
	</tr>
</table>
Human-level error serves as an estimate for Bayes error. With the same training and dev errors, focus on bias and variance tactics accordingly.

- **Avoidable Bias**: The gap between human error (approximation of Bayes error) and training error
	- Improve the training error until it reaches Bayes error, but not beyond to avoid overfitting
- **Variance**: The gap between training and dev errors indicates the variance problem of the algorithm

<br>

<a id="3-3"></a>
### 3-3. Understanding Human-level Performance
Human-level error as a proxy for Bayes error.

**Example: Medical image classification**

suppose:

(a) Typical human - 3% error

(b) Typical doctor - 1% error

(c) Experienced doctor - 0.7% error

(d) Team of experienced doctors - 0.5% error

Bayes error is always less than the lowest value of human-level error. For practical purposes. human-level error can be defined differently, but as a proxy for Bayes error, use the lowest value of human-level error.

<br>

<a id="3-4"></a>
### 3-4. Surpassing Human-level Performance

**Problems where ML significantly surpasses human-level performance:**
- online advertising
- product recommendations
- logistics (predicting transit time)
- loan approvals

These are typically structured data problems, not natural perception problems like computer vision, speech recognition, and natural language processing, where ML can better find statistical patterns than humans. 

In some natural perception tasks, computers have surpassed human-level performance with enough data, although it's challenging for computers to reach this level.

<br>

<a id="3-5"></a>
### 3-5. Improving your Model Performance
Let's pull it together into a single guideline on how to improve the performance of your learning algorithm.

- Orthogonalization
- Set up dev and test sets
- Human-level performance as a proxy for Bayes error
- How to estimate your avoidable bias and variance

**Two Fundamental Assumptions of Supervised Learning**:

1. You can fit the training set pretty well ($\approx$ low avoidable bias)
2. The training performance generalizes pretty well to dev/test sets ($\approx$ low variance)

Use different strategies to address bias and variance issues:

**Reducing (Avoidable) Bias and Variance**
<table>
	<tr>
		<th>Issue</th>
		<th>Solution Ideas</th>
	</tr>
	<tr class="1">
		<td align="center">Avoidable Bias<br>(human error - training error)</td>
		<td>- Train a bigger model<br>
- Train longer/Improve optimization algorithms(momentum, RMSprop, Adam)<br>
- Experiment with NN architecture/hyperparameter</td>
	</tr>
	<tr class="4">
		<td align="center">Variance <br>(training error - dev error)</td>
		<td>- Get more data<br> - Apply regularization (L2, dropout, data augmentation)<br>
- Experiment NN architecture/hyperparameter</td>
	</tr>
</table>