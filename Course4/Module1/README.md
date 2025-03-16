
# Module1 - Foundations of Convolutional Neural Networks

### Summary
> Implement the foundational layers of CNNs (pooling, convolutions) and stack them properly in a deep network to solve multi-class image classification problems.
> - Learning Objectives
>   - Explain the convolution operation
>   - Apply two different types of pooling operations
>   - Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
>   - Build a convolutional neural network
>   - Implement convolutional and pooling layers in numpy, including forward propagation
>   - Implement helper functions to use when implementing a TensorFlow model
>   - Create a mood classifier using the TF Keras Sequential API
>   - Build a ConvNet to identify sign language digits using the TF Keras Functional API
>   - Build and train a ConvNet in TensorFlow for a binary classification problem
>   - Build and train a ConvNet in TensorFlow for a multiclass classification problem
>   - Explain different use cases for Sequential and Functional APIs

### Table of Contents
1. [Convolutional Neural Networks](#1)
	- 1-1. [Computer Vision](#1-1)
	- 1-2. [Edge Detection](#1-2)
	- 1-3. [Padding](#1-3)
	- 1-4. [Strided Convolutions](#1-4)
	- 1-5. [Convolutions Over Volume](#1-5)
	- 1-6. [One Layer of a Convolutional Network](#1-6)
	- 1-7. [Simple Convolutional Network Example](#1-7)
	- 1-8. [Pooling Layers](#1-8)
	- 1-9. [CNN Example](#1-9)
	- 1-10. [Why Convolutions?](#1-10)



<a id="1"></a>
## 1. Convolutional Neural Network
<a id="1-1"></a>
### 1-1. Computer Vision
Computer Vision tasks often tackled by convolutional neural networks (CNNs) include:
- **Image Classification**: Assigning a label to an entire image
- **Object Detection**: Identifying objects within an image and their locations
- **Neural Style Transfer**: Applying artistic style of one image to the content of another

**Challenge with large images:**

Larger images result in more parameters, which increases the risk of overfitting and requires more computational resources

**Convolution Operation:**

A fundamental building block of convolutional neural networks, allowing efficient processing of large images.

<br>

<a id="1-2"></a>
### 1-2. Edge Detection 
Filters of Edge Detection
<table>
	<tr>
		<th>Vertical Filter</th>
		<th>Horizontal Filter</th>
		<th>Sobel Filter</th>
		<th>Scharr Filter</th>
	</tr>
	<tr>
		<td>
		<table>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
</table>
		</td>
		<td>
		<table>
	<tr>
		<td>1</td>
		<td>1</td>
		<td>1</td>
	</tr>
	<tr>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
		<td>-1</td>
		<td>-1</td>
		<td>-1</td>
	</tr>
</table>
		</td>
<td>
		<table>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
	<tr>
		<td>2</td>
		<td>0</td>
		<td>-2</td>
	</tr>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
</table>
		</td>
		<td>
		<table>
	<tr>
		<td>3</td>
		<td>0</td>
		<td>-3</td>
	</tr>
	<tr>
		<td>10</td>
		<td>0</td>
		<td>-10</td>
	</tr>
	<tr>
		<td>3</td>
		<td>0</td>
		<td>-3</td>
	</tr>
</table>
		</td>
	</tr>
</table>

**Learning Filters:**

Neural networks can learn filters (like edge detectors) during training, improving over hand-coded filters by computer vision researchers

Example:

<table style="border-collapse:collapse; border:none;">
	<tr>
		<th >Input Image (6 x 6)</th>
		<th></th>
		<th>Vertical Filter (3 x 3)</th>
		<th></th>
		<th>Output (4 x 4)</th>
	</tr>
	<tr>
		<td>
			<table>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
	<tr>
		<td>10</td>
		<td>10</td>
		<td>10</td>
		<td>0</td>
		<td>0</td>
		<td>0</td>
	</tr>
</table>
		</td>
		<td>*</td>
		<td>
		<table>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
	<tr>
		<td>1</td>
		<td>0</td>
		<td>-1</td>
	</tr>
</table> 
		</td>
		<td>=</td>
		<td>
			<table>
	<tr>
		<td>0</td>
		<td>30</td>
		<td>30</td>
		<td>0</td>
	</tr>
	<tr>
		<td>0</td>
		<td>30</td>
		<td>30</td>
		<td>0</td>
	</tr>
	<tr>
	<tr>
		<td>0</td>
		<td>30</td>
		<td>30</td>
		<td>0</td>
	</tr>
	<tr>
		<td>0</td>
		<td>30</td>
		<td>30</td>
		<td>0</td>
	</tr>
</table>	
		</td>
	</tr>
</table>

<br>

<a id="1-3"></a>
### 1-3. Padding
Padding helps maintain spatial dimensions of the input

**Dimensions**

- Input image: $(n, n)$
- Filter: $(f, f)$
- Output: $(n - f + 1, n - f + 1)$

**Two downsides without padding:**

1. **Shrinking output size**: Every time you apply a convolutional operator, the dimension shrinks
2. **Loss of edge information**: The pixels of the corners are touched fewer times than the pixels that are relatively in the middle of the image

To avoid these downsides, you can pad the image with additional borders of pixels.

**Output Dimension**

$$(n - f + 2p + 1, n - f + 2p + 1)$$

**Types of Convolutions**
- **Valid Convolution**: No padding

$$(n, n) * (f,f) \rightarrow (n - f + 1, n - f + 1)$$
 
- **Same Convolution**: Pad so that output size is the same as the input size

$$(n + 2p, n + 2p) * (f, f) \rightarrow (n - f + 2p + 1, n - f + 2p + 1)$$

By convention, filter size is odd.


**Padding Calculation**:

$$p = \frac {f - 1} {2}$$

<br>

<a id="1-4"></a>
### 1-4. Strided Convolutions
Stride controls the step size of the convolution filter over the input.

- If the fraction is not an integer, we round down
- If a part of the filter hangs outside of the input image, we don't do the computation
- The filter must lie entirely within the image or the image plus the padding region

**Output Dimension**

$$(\left\lfloor \frac {n - f + 2p} {s} + 1\right\rfloor, \left\lfloor \frac {n - f + 2p} {s} + 1\right\rfloor)$$

<br>

<a id="1-5"></a>
### 1-5. Convolutions Over Volume
**Convolutions on RGB images**

The filter must match the number of input channels, producing a 2D output per filter.


**Multiple filters**

Applying multiple filters to detect each corresponding feature produces a multi-channel output.

**Output Dimension**

$$(n + 2p, \space n + 2p, \space n_c) * (f, f, n_c) \rightarrow (\left\lfloor \frac {n - f + 2p} {s} + 1\right\rfloor, \space \left\lfloor \frac {n - f + 2p} {s} + 1\right\rfloor, \space n_c^{'})$$

where $n_c$ is the number of input and filter's channels and $n_c^{'}$ is the number of filters

<br>

<a id="1-6"></a>
### 1-6. One Layer of Convolutional Network

The output will be added with a bias of the same dimension and then passed by an activation function.

**Summary of notation**

If layer $l$ is a convolution layer:
- $f^{[l]}$: filter size
- $p^{[l]}$: padding
- $s^{[l]}$: stride
- $n_c^{[l]}$ = number of filters
- Each filter is: $(f^{[l]}, f^{[l]}, n_c^{[l-1]})$
- Activations:
	- $a^{[l]} \rightarrow (n_H^{[l]}, n_W^{[l]}, n_c^{[l]})$
	- $A^{[l]} \rightarrow (m, n_H^{[l]}, n_W^{[l]}, n_c^{[l]})$
- Weights: $(f^{[l]}, f^{[l]}, n_c^{[l-1]}, n_c^{[l]})$
- Bias: $n_C^{[l]} - (1, 1, 1, n_c^{[l]})$
- Input: $(n_H^{[l-1]}, n_W^{[l-1]}, n_c^{[l-1]})$
- output: $(n_H^{[l]}, n_W^{[l]}, n_c^{[l]})$

where $n_H^{[l]} = \left\lfloor \frac {n_H^{[l-1]} + 2p^{[l]} -f^{[l]}} {s^{[l]}} + 1\right\rfloor, \space n_W^{[l]} = \left\lfloor \frac {n_W^{[l-1]} + 2p^{[l]} -f^{[l]}} {s^{[l]}} + 1\right\rfloor$

<br>

<a id="1-7"></a>
### 1-7. Simple Convolutional Network Example
Types of layers in a convolutional network:
- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)

<br>

<a id="1-8"></a>
### 1-8. Pooling Layers
Pooling reduces the spatial dimensions of the input.

**Types of Pooling**:
- Max pooling
- Average pooling

**Hyperparameters**
- f: filter size
- s: stride

pooling layer doesn't have parameters but hyperparameters


**Output Dimension**

$$(n_H , n_W, n_c) \rightarrow (\left\lfloor \frac {n_H - f} {s} + 1\right\rfloor, \space \left\lfloor \frac {n_W - f} {s} + 1\right\rfloor, \space n_c)$$

<br>

<a id="1-9"></a>
### 1-9. CNN Example
Neural network example:
|Layer|Activation shape|Activation Size|# parameters|
|--|--|--|--|
|Input|(32, 32, 3)|3,072|0|
|CONV1 (f=5, s=1)|(28, 28, 6)|4,704|456|
|POOL1|(14, 14, 6)|1,176|0|
|CONV2 (f=5, s=1)|(10, 10, 16)|1,600|2,416|
|POOL2|(5, 5, 16)|400|0|
|FC3|(120, 1)|120|48,120|
|FC4|(84, 1)|84|10,164|
|Softmax|(10, 1)|10|850|

This neural network is inspired by the LeNet-5 architecture. Pooling layers don't have parameters and most of the parameters are present in fully-connected layers. 

The activation size tends to go down gradually as you go deeper in the neural network, if it drops too quickly, it's usually not good for performance as well.

A lot of convNets will have properties and patterns similar to this.

<br>

<a id="1-10"></a>
### 1-10. Why Convolutions?

There are two main advantages of convolutional layers over fully-connected layers


- **Parameter sharing**:  A feature detector (such as a vertical edge detector) that is useful in one part of the image is probably useful in another part of the image.
- **Sparsity of connections**: In each layer, each output value depends only on a small number of inputs.


**Number of parameters**
- $f^{[l]}$ is the filter height and width
- $n_c^{[l-1]}$ is the number of channels in the previous layer
- $n_c^{[l]}$ is the number of channels in the current layer
- The "1" is the bias term

$$(f^{[l]} \times f^{[l]} \times n_c^{[l-1]} + 1) \times n_c^{[l]}$$