# Module2 - ML Strategy
### Summary
> Develop time-saving error analysis procedures to evaluate the most worthwhile options to pursue and gain intuition for how to split your data and when to use multi-task, transfer, and end-to-end deep learning.
> - Learning Objectives
>    - Describe multi-task learning and transfer learning
>    - Recognize bias, variance and data-mismatch by looking at the performance of your algorithm on the train/dev/test sets

### Table of contents
1. [Error Analysis](#1)
	- 1-1. [Carrying Out Error Analysis](#1-1)
	- 1-2. [Cleaning Up Incorrectly Labeled Data](#1-2)
	- 1-3. [Build your First System Quickly, then Iterate](#1-3)
2. [Mismatched Training and Dev/Test Set](#2)
	- 2-1. [Training and Testing on Different Distributions](#2-1)
	- 2-2. [Bias and Variance with Mismatched Data Distributions](#2-2)
	- 2-3. [Addressing Data Mismatch](#2-3)
3. [Learning From Multiple Tasks](#3)
	- 3-1. [Transfer Learning](#3-1)
	- 3-2. [Multi-task Learning](#3-2)
4. [End-to-End Deep Learning](#4)
	- 4-1. [What is End-to-End Deep Learning?](#4-1)
	- 4-2. [Whether to use End-to-End Deep Learning](#4-2)

<a id="1"></a>
## 1. Error Analysis

<a id="1-1"></a>
### 1-1. Carrying Out Error Analysis
To evaluate the effectiveness of potential improvements to your model, perform error analysis on dev set. Here is how you can systematically do it:

**Loot at dev examples to evaluate ideas**

Example: Cat classifier 
- Accuracy 90%, Error 10%

If there are some dog images in the mislabeled examples, Should you try to make your cat classifier do better on dogs?

**Steps for Error Analysis**
1. Identify mislabeled examples

- Collect ~100 mislabeled dev set examples

3. Categorize errors
- For each mislabeled example, categorize the type of error. For instance, if dogs are mislabeled as cats, identify them explicitly.

4. Quantify error types
- Calculate the percentage of each error type. For example, if 5 out of 100 errors are due to dogs being mislabeled, it is 5% dog problem.

If you solve this dog problem the error rate will go down from 10% to 9.5%. It gives you the ceiling, upper bound on how much you could improve performance by working on the dog problem.

If there are 50 dog images out of the 100 mislabeled examples, then the error would down from 10% to 5%.

**Evaluate multiple ideas in parallel**
Ideas for cat detection:
- Fix pictures of dogs being recognized as cats
- Fix great cats (lions, panthers, etc..) being misrecognized
- Improve performance on blurry images

|Image|Dog|Great Cat|Blurry|Instagram|Comments|
|--|--|--|--|--|--|
|1|O|||O|Pitball|
|2|||O|O||
|3||O|O||Raining day at zoo|
|...|...|...|...|...|...|
|% of total|8%|42%|62%|12%|

Based on the errors above, it is reasonable to focus on the most frequent error types for the highest impact on improvements and you can also work on several error types being dedicated to each team.

This process will help prioritization decisions and how promising different approaches to work on.

<br>

<a id="1-2"></a>
### 1-2. Cleaning Up Incorrectly Labeled Data

**Training Set**

DL algorithms are quite robust to random errors in the training set. Random errors or near random errors are usually acceptable but sensitive to Systematic errors, for example, all of the white dogs are labeled as cats so the classifier will classify white dogs as cats.


**Dev Set**

Example:
|Image|Dog|Great Cat|Blurry|Incorrectly labeled|Comments|
|--|--|--|--|--|--|
|...|...|...|...|...|...|
|98||||O|Labeler missed cat in background|
|99|O||||Pitball|
|100||||O|Drawing of a cat;<br> not a real cat|
|% of total|8%|43%|61%|6%|

Error Analysis
|Error|Senario 1|Senario 2|
|--|--|--|
|Overall dev set error|10%|2%|
|Errors due to incorrect labels|0.6%|0.6%|
|Errors due to other causes|9.4%|1.4%


If the error due to incorrect labels is relatively less than the erros due to other causes in error analysis, it is advisable to focus rather on the higher error problem.

If the error due to incorrect labels is relatively higher than 30%, it seems worthwhile to fix up the incorrect labels in the dev set.

Goal of dev set is to help you select between classifier A and B
If there are two classifiers A with 2.1% error and B with 1.9% error, since the error due to incorrect labels is 0.6%, you can't tell which one is better.


**Correcting incorrect dev/test set examples**
- Apply the same process to your dev and test sets to make sure they continue to come from the same distribution.
- Consider examining examples your algorithm got right as well as ones it got wrong
- Train and dev/test sets may now come from slightly different distributions

<br>

<a id="1-3"></a>
### 1-3. Build your First System Quickly, then Iterate
Speech recognition example
- Noisy background
	- Cafe noise
	- Car noise
- Accented speech
- Far from microphone
- Young children's speech
- Stuttering


Recommendation
- Set up dev/test set and metric
- Build the initial system quickly
- Use Bias/Variance analysis & Error analysis to prioritize next steps

<br>


<a id="2"></a>
## 2. Mismatched Training and Dev/Test Set
<a id="2-1"></a>
### 2-1. Training and Testing on Different Distributions

**Cat app example:** 
||Train|Dev/Test|
|--|--|--|
|Data from|Webpages|Mobile app|
|Data size|200,000|10,000|

Division example:

|Train|Dev|Test|
|--|--|--|
|200k (webpages)|5k|5k|
|200k + 10k (mobile app)|2.5k|2.5k|

**Speech recognition example:**

||Train|Dev/Test|
|--|--|--|
|Data from|- Purchased data<br> - Smart speaker control<br>- Voice keyboard|- Speech activated rearview mirror|
|Data size|500,000|20,000|

Division example:

|Train|Dev|Test|
|--|--|--|
|500k (train)|10k|10k|
|500k + 10k(dev/test)|5k|5k|

If the train and dev/test sets come from different distributions, it is better to assign all the train set and some portion of dev/test set to it, rather than shuffling them.

<br>

<a id="2-2"></a>
### 2-2. Bias and Variance with Mismatched Data Distributions
**Cat classifier example:**

Assume humans get $\approx$ 0% error
- Training error: 1%
- Dev error: 10%

When the distributions of train and dev are not the same, the error difference between training and dev above doesn't mean that it has a variance problem, but it reflects that dev set contains images that are more difficult to classify accurately.

**Training-dev set**: Same distribution as training set but not used for training

|Error|Senario 1|Senario 2|Senario 3|Senario 4|
|--|--|--|--|--|
|Human error|0%|0%|0%|0%|
|Training error|1%|1%|10%|10%|
|Training-dev error|9%|1.5%|11%|11%|
|Dev error|10%|10%|12%|20%|
|Analysis|High Variance|Data Mismatch|High Bias|High Bias<br>Data Mismatch|


**Bias/variance on mismatched training and dev/test sets**

Human-level error 

$\qquad\updownarrow\text{avoidable bias}$

Training error 

$\qquad\updownarrow\text{variance}$

Training-dev error 

$\qquad\updownarrow\text{data mismatch}$

Dev error 

$\qquad\updownarrow\text{degree of overfitting to the dev set}$

Test error

**More general formulation**

|Error|General speech recognition |Rearview mirror speech data|
|--|--|--|
|Human-level|Human-level|
|Error on examples trained on|Training error|
|Error on examples not trained on|Training-dev error|Dev/Test error|

<br>

<a id="2-3"></a>
### 2-3. Addressing Data Mismatch
- Carry out manual error analysis to try to understand the difference between training and dev/test sets
- Make training data more similar, or collect more data similar to dev/test sets

**Artificial data synthesis**

be cautious whether you might be accidentally simulating data only from a tiny subset of the space of all possible examples.

<br>


<a id="3"></a>
## 3. Learning From Multiple Tasks

<a id="3-1"></a>
### 3-1. Transfer Learning
Transfer learning involves taking a pre-trained neural network designed for one task and adapting it to another, typically related, task. This approach is useful when you have a large amount of data for the source task and relatively less data for the target task.

Examples:

- Image Recognition -> Radiology Diagnosis: Use a pre-trained image recognition model, modify the output layer for radiology, and fine-tune it with radiology data

- Speech Recognition -> Wakeword/Triggerword Detection: Adapt a speech recognition model to detect specific wakewords

**When does transfer learning make sense?**

- Transfer learning makes sense when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to

Transfer from Task A to Task B
- Task A and B have the same input $x$
- You have a lot more data for Task A than Taks B
- Low-level features from Task A could be helpful for learning Task B


#### keywords
- pre-training: Train a neural network on a large dataset for the source task
- fine-tuning: Adapting the pre-trained model to the target task with less data

<br>

<a id="3-2"></a>
### 3-2. Multi-task Learning
Multi-task learning involves training a model on multiple tasks simultaneously, leveraging shared representations. This approach is beneficial when tasks share common features.

Simplified autonomous driving example:
- Tasks: Detecting pedestrians, cars, stop signs, and traffic lights
- Input: One image $x^{(i)}$ with multiple labels

loss : 

$$y{(i)} = \frac {1} {m} \sum_{i=0}^{m} \sum_{j=0}^{4} L(\hat y^{(i)}_j, y^{(i)}_j)$$

where $L$ is the usual logistic loss


**Difference with softmax regression:**
- Multi-task Learning: Multiple labels per image
- Softmax Regression: Single lable per image

**When does multi-task learning make sense?**

- Training on a set of tasks that could benefit from having shared lower-level features
- Usually, the amount of data you have for each task is quite similar
- Can train a big enough neural network to do well on all the tasks
<br>


<a id="4"></a>
## 4. End-to-End Deep Learning

<a id="4-1"></a>
### 4-1. What is End-to-End Deep Learning?
End-to-end deep learning replaces multiple stages of processing with just a single neural network, directly mapping input to output.

Speech recognition example:

**Traditional pipeline**:

audio(x) $\xrightarrow{MFCC}$ features $\xrightarrow{ML}$ phonemes $\xrightarrow{}$ words $\xrightarrow{}$ transcript (y)

**End-to-end approach**:

audio $\xrightarrow{}$ transcript

This approach eliminates intermediate steps and can be highly effective with sufficient data

<br>

<a id="4-2"></a>
### 4-2. Whether to use End-to-End Deep Learning

**Pros and cons of end-to-end deep learning**
- Pros:
	- Let the data speak
	- Less hand-designing of components needed
- Cons:
	- May need a large amount of data
	- Excludes potentially useful hand-designed components

**Applying end-to-end deep learning**

Key question: Do you have sufficient data to learn a function of the complexity needed to map $x$ to $y$?

Using end-to-end deep learning depends on the availability of data and the complexity of the task. If you have sufficient data and the task benefits from a direct input-output mapping, end-to-end learning can be advantageous.