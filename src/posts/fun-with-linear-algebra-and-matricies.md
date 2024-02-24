---
title: "Fun with Linear Algebra and Matricies"
description: "Exploring matrix mathmatical operations..."
tags: ["Math"]
date: "02-26-2024"
link: "/posts/fun-with-linear-algebra-and-matricies"
---

As you can probably tell by my <a href="https://www.zacmessinger.com/posts/gradient-descent-scratch" target="_blank">last post</a>, I've spent a lot of time recently trying to better understand the core mathmatical principles used in machine learning, and to say the least, it's been quite the endevour. Relearning the fundamentals of Linear Algebra, Calculus, Statistics, & Probability has felt like an intellectual rollercoaster, where everytime I've reached the top of a hill, a steep drop into an even more complicated subject matter quickly follows! In the words of my inner voice (which sounds strangely like Master Yoda):

_Mastered matrix multiplication, have you? Try eisenvectors!_

My primary educational resource on this quest has been Coursera's [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science) online course taught by Louis Serrano. While parts of it have been highly challenging, overall it's been an immensily helpful resource that I would recommend to anyone looking to sharpen their mathmatical skillets. I've found that learning the libraries, frameworks, and processes of data science can only take you so far if you don't have a strong grasp of the underlying mathmatical concepts, especially when it comes to deep learning.

That brings us to the topic of today's post, which is going to focus on exploring a few linear algebra operations for some of the building blocks of machine and deep learning - scalers, vectors, and matricies! Let's dive into it.

### Library Setup.

As always, let's begin by importing the libraries we will use for the remainder of this post, which today will only be [PyTorch](https://pytorch.org/). PyTorch is a dynamic framework developed by Meta for building deep learning models such as ChatGPT, and while we won't be building any models with PyTorch today, working with matricies is a core part of it's functionality.

```python
import torch
```

It's worth noting that there are some other fantastic mathmatical and deep learning libraries out there for working with matricies, with two of the bigger ones being [Numpy](https://numpy.org/) and [TensorFlow](https://www.tensorflow.org/) respectively. I would encourage you to check them out if you are interested in learning more about them.

### The Building Blocks

To start, let's first define what scalers, vectors and matricies are before exploring how we can use them to perform linear algebraic operations.

-   **Scalers**: A scalar is a single number that has magnitude (i.e size), but no direction, which is just a fancy way of saying it is a single number with zero dimensions. Scalers earn their name because they are often used to "scale" a matrix up or down via multiplication. As an example, $5$, $-15$, and $42$ are all scalers under this definition.
-   **Vectors**: Vectors are one dimensional objects that contain both magnitude and direction, and are often used to describe physical phenomena like velocity, acceleration, force, and position. Programtically, they are expressed as arrays of lists with ordered indicies such as $[1,2,3]$ and $[5, -30, 20]$.
-   **Matricies**: Matrices are two-dimensional arrays of numbers, consisting of rows and columns, which can be used to represent a variety of data and are prominantly used in Linear Algebra to represent and solve systems of linear equations.. Programatically, they are expressed as an array of arrays such as $[[1,2,3], [4,5,6]]$ and $[[1,2],[3,4],[5,6],[7,8]]$.

### Key Operations

There are several key operations that can be performed on matricies, each with their own set of rules and operations. For the purposes of this post, we are mainly going to focus on a few of the core ones related to machine and deep learning, specifically:

-   Addition and Subtraction
-   Scaler Multiplication
-   Element-Wise Multiplication
-   Dot Product Multiplication
-   Transposition

I've ranked these operations by order of complexity, so if you feel confident with one topic, feel free to skip to the next one. Hopefully the first few act act as more of a general refresher!

### Addition and Subtraction

Let's start with the easiest operation, addition and subtraction. Adding or subtracing two matricies together requires that the matricies have the same number of rows and columns, where a new matrix is produced by adding or subtracting the corresponding elements from each matrix.

As an example, let's say we have two matricies, $A$ and $B$:

$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} 
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$

Which progrmatically, can be expressed in PyTorch as:

```python
# Matrices
A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])
```

If we wanted to add matrix $A$ to matrix $B$, we could so by performing the folowing element wise operations:

$
C = \begin{pmatrix}
1 + 5 & 2 + 6 \\
3 + 7 & 4 + 8
\end{pmatrix}
$

$
C = \begin{pmatrix}
6 & 8 \\
10 & 12
\end{pmatrix}
$

Simple enough, right? It's even easier in PyTorch!

```python
A.add(B)
```

```
tensor([[ 6,  8],[10, 12]])
```

For the sake of brevity, we'll skip subtraction because it follows the same pattern. Onto Scaler Multiplcation!

### Scaler Multiplication

While adding and subtracting matricies from each other tend to be more use case dependent, scaler multiplication is a fundamental operation in both machine learning and deep learning that is used in a variety of contexts. Just to name a few:

-   **Feature Scaling**: Scalar multiplication is often used to scale features in a dataset to a consistent range, since training a model on a feature set of different numerical scales can negatively impact the resulting models performance and efficiency. Two of the more common feature scaling techniques that include scaler multiplication are _normalization or Min-Max Scaling_, which involves rescaling every datapoint between 0 and 1, and _standardization or Z-Score normalization_, which involves rescaling every data point to within a single standard deviation around a mean of 0
-   **Gradient Descent**: A crucial part of gradient descent involves multiplying our derived gradients by a learning rate in order to determine the size of a step to take in attempting to decrease our cost function towards a local minimum. Most notably, gradient descent is a crucial part of training deep neural networks during what is known as the [backpropogation process](https://en.wikipedia.org/wiki/Backpropagation), which is a topic onto itself that I hope to explore in a future blog post.

Scaler multiplication is relatively straightforward process to carry out - we simply multiply every element in a matrix or vector by the provided scaler, which as a reminder, is just a single number.

Let's now look at an example by multipling matrix $A$ by a newly defined scaler $C$:

$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} 
B = \begin{pmatrix}
3
\end{pmatrix}
$

$
C = \begin{pmatrix}
1*3 & 2*3 \\
3*3 & 4*3
\end{pmatrix} 
$

$
C = \begin{pmatrix}
3 & 6 \\
9 & 12
\end{pmatrix} 
$

```python
A = torch.tensor([[1,2], [3,4]])
C = 3

print(A * C)
```

```
tensor([[ 3,  6], [ 9, 12]])
```

With that, we're almost through the easy topics. Let's move onto element wise multiplication!

### Element Wise Multiplication

Similar to matrix addition and subtraction, element wise multiplication is performed by multiplying each element in a provided matrix by it's counterpart in a corresponding matrix. It's commonly used to perform a variety of tasks within machine learning and deep learning, but most notably:

-   **Feature Engineering**: When creating new features from existing ones, element-wise multiplcation can be used to generate interaction items between disparate features that individually may not have a high correlation with the target label, but together potentially do. For example, we may find that multiplying the number of bedrooms by the number of bedrooms in a housing dataset into a new feature called `num_bed_bath` may have a higher correlation with predicting the price of a house than the raw number of bedrooms or bathrooms do invidually.
-   **Activation Functions**: While this is again a slightly deeper topic that is beyond the scope of this post, an activation function is a function that can be applied via element-wise multiplication to the output of a given layers neurons that allows neural networks to model non-linear data. As an exampe, the ReLU family of activation functions (e.g ReLU, Leaky ReLU, and Parametric ReLU) are commonly used in deep learning for this purpose as well as for mitigiating what is known as the vanishing gradient problem.

Let's go over a simple example of how to perform element-wise multiplcation with the matricies $A$ and $B$ that we defined previously.

$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} 
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$

$
C = \begin{pmatrix}
1*5 & 2*6 \\
3*7 & 4*8
\end{pmatrix} 
$

$
C = \begin{pmatrix}
5 & 12 \\
21 & 32
\end{pmatrix} 
$

```python
# Matrices
A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])

A.mul(B)
```

```
tensor([[ 5, 12],[21, 32]])
```

### Dot Product Matrix Mulitplication

With the easy operations out of the way, let's start diving into the slightly more complex ones starting with _dot product multiplication_. The uses of dot production multiplication are almost too numerous to list out, but to highlight a couple especially important ones:

-   **Gradient Descent**: Dot product multiplication is used in linear regregression to generate new prediction values after a single step of gradient descent by multiplying our features (e.g `X`) values by our newly calculated gradient values for each feature. It is again used in actually calculating the gradients for each feature, where we multiply the differences between our predicted values and actual `y` values by what is known as the _transpose_ of our `X` values. More on transpososition in a sec!
-   **Neural Network Forward Pass**: During what is known as the _foward pass_ part of the training process in a neural network, dot product multiplication is used to compute the weighted sum for each neuron in a given layer by multiplying the input features or `X` values of the dataset by the vector of the neuron's current weights. This weighted sum is then usually passed to an activation function via scaler multiplication, as we noted previously above.

Mathmatically, dot product multiplication is similar to element-wise multiplication, but instead of multiplying each element in a provided by matrix by it's counterpart in a corresponding matrix, we multiply each row in the first matrix by each column of the second matrix before adding them together to produce the result matrix. Unlike most of our previous matrix operations, Dot Product multiplication can not always be be performed on matricies of equal size - the constraint is that the number of columns in the first matrix must equal the number of columns in the second matrix.

This might seem confusing, so let's go through a couple example starting with two matricies of identical rows and columns:

$
A = \begin{pmatrix}
1 & 2 \\
3 & 4 
\end{pmatrix} 
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$

$
C = \begin{pmatrix}
(1 * 5) + (2 * 7) & (1 * 6) + (2 * 8) \\
(3 * 5) + (4 * 7) & (3 * 6) + (4 * 8)
\end{pmatrix}
$

$
C = \begin{pmatrix}
19 & 22 \\
43 & 50
\end{pmatrix}
$

While a little more involved than element-wise multiplication, the resulting math for dot product multiplication is still not too bad, although we can imagine that the larger our matricies become, the more involved our calculations would be. Fortunately, PyTorch makes this operation incredibly easy so we don't have to actually worry about what's going on underneath the hood when trainig a model:

```python
# Matrix
A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])

torch.matmul(A, B)
```

```
tensor([[19, 22],[43, 50]])
```

Let's now look at dot product multiplication through the lens of multiplying a vector by a vector, as highlighted in the forward pass process of training a neural network. Let's imagine that the elements in vector `A` represents the differnt `X` values in a feature set, while the elements in vector `B` represent the current weights assigned to the those `X` values in single a hypothetical neuron:

$
A = \begin{pmatrix}
1 & 2 & 3
\end{pmatrix} 
B = \begin{pmatrix}
1 \\
1.5 \\ 
2 \\
\end{pmatrix}
$

Because the number of columns in vector `A` equal the number of rows in vector `B`, we can proceed with dot product multiplication in order to calculate a weigted sum for the neuron!

$
C = \begin{pmatrix}
(1*1) + (2*1.5) + (3 * 2)
\end{pmatrix} 
$

$
C = \begin{pmatrix}
1 + 3 + 6
\end{pmatrix} 
$

$
C = 10
$

Pretty cool, huh? Finally, let's look at what at happens if we try dot product multiplication on two matricies whose columns and rows don't align. A perfect example of this can be found in the gradient descent algorithm, which as a reminder, is expressed by the formula:

$\theta*j := \theta_j - \alpha \frac{1}{m} \sum*{i=1}^{m} (h\_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}\$

(_Note: If you need a refresher on how the gradient descent algorithm works, checkout [my last post](https://www.zacmessinger.com/posts/gradient-descent-scratch)_!)

Let's focus on the part of gradient descent where we measure the total cost associated for each of our current $theta$ values, which is calulcated by applying dot product multilication on the vector of the differences between our predicted $y$ values and actual $y$ values, and the matrix `X` of our input features. Suppose the differences between our predicted and actual $y$ our represented by the vector $D$, while our input features are represented by the matrix $X$:

$
D = \begin{pmatrix}
-2 \\
-4 \\
-6 \\
-8 \\
\end{pmatrix}  
X = \begin{pmatrix}
1 & 1 \\
1 & 2 \\
1 & 3 \\
1 & 4
\end{pmatrix}
$

Because vector $D$ has $1$ column while while matrix $X$ has 4 rows, we cannot perform dot production multiplication. Don't believe me? Let's try it out in PyTorch:

```python
y = torch.tensor([[-2], [-4], [-6], [-8]])
X = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4]])

y.matmul(X)
```

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1 and 4x2)
```

So what are we to do here? Let's move into our next matrix operation, _transposition_, to find out how to solve this problem.

### Transposition

Matrix transposition is a mathmatical operation that involves flipping a matrix over it's diagonal by exchanging it's rows and columns, meaning the new resulting matrix has the same number of columns as the initial matrix had rows. The easiest way to demonstrate this is visually, so let's go over a quick example using matrix $A$ as defined below:

$
A = \begin{pmatrix}
a & b\\
c & d \\
e & f \\
\end{pmatrix}
$

The transpose of matrix A, expressed as $A^T$, would be the following:

$
A^T = \begin{pmatrix}
a & c & e\\
b & d & f
\end{pmatrix}
$

This seems like it would be pretty helpful for our initial problem statement above, where we could not perform dot product multiplication for vector $D$ and matrix $X$ because the number of columns in vector $D$ did not match the number of rows in matrix $A$. The solution lies in transposition - what if we tranpose vector $D$, so that instead of having a `4x1` shape it now has a `1x4` shape? Let's see what the looks like:

$
D = \begin{pmatrix}
-2 \\
-4 \\
-6 \\
-8 \\
\end{pmatrix}
$

$
D^T = \begin{pmatrix}
-2 & -4 & -6 & - 8
\end{pmatrix}
$

Perfect! Now we can perform dot product multiplciation to finish calculating the vector of total costs for our gradient descent algorithm:

$
D^T = \begin{pmatrix}
-2 & -4 & -6 & - 8
\end{pmatrix}
X = \begin{pmatrix}
1 & 1 \\
1 & 2 \\
1 & 3 \\
1 & 4
\end{pmatrix}
$

$
D^TX = \begin{pmatrix}
((-2*1) + (-4*1) + (-6*1) + (-8*1)) & ((-2*1) + (-4*2) + (-6*3) + (-8*4))
\end{pmatrix}
$

$
D^TX = \begin{pmatrix}
(-2 - 4 - 6 - 8) & (-2 - 8 - 18 - 32)
\end{pmatrix}
$

$
D^TX = \begin{pmatrix}
-20 & -60
\end{pmatrix}
$

As we can see, matrix transposition is increcibly useful tool to have in the linear algebra toolbelt, as it is a critical component of not only gradient descent but multiple other algorithms in machine learning. Just as we did with our previous matrix operations, let's now implement this programatically:

```python
def transpose(matrix):
    ans=[[0] * len(matrix) for _ in range(len(matrix[0]))]

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            ans[col][row] = matrix[row][col]

    return ans

A = [[-2], [-4], [-6], [-8]]
transposed = transpose(A)

print(f"A: {A} | Transposed: {transposed}")
print(f"A Dimensions: {torch.tensor(A).shape} | Tranposed Dimensions: {torch.tensor(transposed).shape}")
```

```
A: [[-2], [-4], [-6], [-8]] | Transposed: [[-2, -4, -6, -8]]
A Dimensions: torch.Size([4, 1]) | Tranposed Dimensions: torch.Size([1, 4])
```

Fortunately, we don't have to write out all this code and PyTorch makes it much easier!

```python
y = torch.tensor([[-2], [-4], [-6], [-8]])
X = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4]])

y.T.matmul(X)
```

```
tensor([[-20, -60]])
```

### Conclusion
