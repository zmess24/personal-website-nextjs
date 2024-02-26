---
title: "Fun with Linear Algebra and Matricies"
description: "Exploring matrix mathmatical operations..."
tags: ["Math"]
date: "02-26-2024"
link: "/posts/fun-with-linear-algebra-and-matricies"
---

As you can probably tell by my [last post](https://www.zacmessinger.com/posts/gradient-descent-scratch), I've spent a lot of time recently trying to better understand the core mathematical principles used in machine learning, and to say the least, it's been quite the endeavor. Relearning the fundamentals of Linear Algebra, Calculus, Statistics, & Probability has felt like an intellectual rollercoaster, where every time I've reached the top of a hill, a steep drop into an even more complicated subject matter quickly follows! In the words of my inner voice (which sounds strangely like Master Yoda):

_Mastered matrix multiplication, have you? Try eigenvectors!_

My primary educational resource on this quest has been Coursera's [Mathematics for Machine Learning and Data Science Specialization](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science) online course taught by Louis Serrano. While parts of it have been highly challenging, overall it's been an immensely helpful resource that I would recommend to anyone looking to sharpen their mathematical skill sets. I've found that learning the libraries, frameworks, and processes of data science can only take you so far if you don't have a strong grasp of the underlying mathematical concepts, especially when it comes to deep learning.

That brings us to the topic of today's post, which is going to focus on exploring a few linear algebra operations on some of the building blocks of machine and deep learning - scalars, vectors, and matrices! Let's dive into it.

### Library Setup.

As always, let's begin by importing the libraries we will use for the remainder of this post, which today will only be [PyTorch](https://pytorch.org/). PyTorch is a dynamic framework developed by Meta for building deep learning models such as ChatGPT, and while we won't be building any models with PyTorch, working with matrices is a core part of its functionality.

```python
import torch
```

It's worth noting that there are some other fantastic mathematical and deep learning libraries out there for working with matrices, with two of the bigger ones being [Numpy](https://numpy.org/) and [TensorFlow](https://www.tensorflow.org/). I would encourage you to check them out if you are interested in learning more about them.

### The Building Blocks

To start, let's first define what scalars, vectors and matrices are before exploring how we can use them within linear algebraic operations.

-   **Scalar**: A scalar is a single number that has magnitude (i.e size), but no direction, which is just a fancy way of saying it is a single number with zero dimensions. Scalers earn their name because they are often used to "scale" a matrix up or down via multiplication. As an example, $5$, $-15$, and $42$ are all scalers under this definition.
-   **Vectors**: Vectors are one dimensional objects that contain both magnitude and direction, and are often used to describe physical phenomena like velocity, acceleration, force, and position. Programmatically, they are expressed as arrays or lists with ordered indices such as $[1,2,3]$ and $[5, -30, 20]$.
-   **Matrices**: Matrices are two-dimensional arrays of numbers, consisting of rows and columns, which can be used to represent a complex data and are prominently used in Linear Algebra to represent and solve systems of linear equations. Programatically, they are expressed as an array of arrays such as $[[1,2,3], [4,5,6]]$ and $[[1,2],[3,4],[5,6],[7,8]]$.

### Key Operations

There are several key operations that can be performed on matricies, each with their own set of rules and operations. For the purposes of this post we will focus on a few of the core ones related to machine and deep learning, specifically:

-   Addition and Subtraction
-   Scalar Multiplication
-   Element-Wise Multiplication
-   Dot Product Multiplication
-   Transposition

I've ranked these operations in order of complexity, so if you feel confident with one topic, feel free to skip to the next one. Hopefully the first few act as more of a general refresher.

### Addition and Subtraction

Let's start with the easiest operation, addition and subtraction. Adding or subtracting two matrices together requires that the matrices have the same number of rows and columns, where a new matrix is produced by adding or subtracting the corresponding elements from each matrix.

As an example, let's say we have two matrices, $A$ and $B$:

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

Which programmatically, can be expressed in PyTorch as:

```python
# Matrices
A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])
```

If we wanted to add matrix $A$ to matrix $B$, we could do so by performing the following element-wise operations:

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
tensor([[6, 8],[10, 12]])
```

Many machine learning algorithms, such as gradient descent, use matrix addition and subtraction for updating the parameters of the model iteratively. These updates often involve adding or subtracting matrices that represent gradients or changes to the parameters.

For the sake of brevity, let's skip subtraction as it follows the same pattern of performing element-wise operations and move onto scalar multiplication!

### Scalar Multiplication

Scalar multiplication is a fundamental operation in both machine and deep learning that is used in a variety of contexts. To name a few:

-   **Feature Scaling**: Scalar multiplication is often used to scale features in a dataset to a consistent range, since training a model on feature sets of different numerical scales can negatively impact the resulting models performance and efficiency. Two of the more common feature scaling techniques that include scalar multiplication are _normalization or Min-Max Scaling_, which involves rescaling every datapoint between 0 and 1, and _standardization or Z-Score normalization_, which involves rescaling every data point to within a single standard deviation around a mean of zero.
-   **Gradient Descent**: A crucial part of gradient descent involves using scalar multiplication to multiply our derived gradients by a learning rate in order to determine the size of the step to take in attempting to decrease our cost function towards a local or global minimum. Most notably, gradient descent is a crucial part of training deep neural networks during what is known as the [back propagation process](https://en.wikipedia.org/wiki/Backpropagation), which is a topic onto itself that I hope to explore in a future blog post.

Scalar multiplication is a relatively straightforward process to carry out. We simply multiply every element in a matrix or vector by the provided scaler, which as a reminder, is just a single number.

Let's look at an example by multiplying matrix $A$ by scaler $C$:

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

Again, this process is quite simple to carry out in PyTorch.

```python
A = torch.tensor([[1,2], [3,4]])
C = 3

print(A * C)
```

```
tensor([[3, 6], [9, 12]])
```

With that, let's move onto element-wise multiplication, which is the last of our easy topics!

### Element Wise Multiplication

Much like matrix addition and subtraction, element-wise multiplication is performed by multiplying each element in a provided matrix by its counterpart in a corresponding matrix. It's commonly used to perform a variety of tasks within machine and deep learning such as:

-   **Feature Engineering**: When creating new features from existing ones, element-wise multiplication can be used to generate interaction items between disparate features that individually may not have a high correlation with the target label, but together potentially do. For example, we may find that multiplying the number of bedrooms by the number of bathrooms in a housing dataset into a new feature called `num_bed_bath` may have a higher correlation with predicting the price of a house than the raw number of bedrooms or bathrooms do individually.
-   **Activation Functions**: While this is a slightly deeper topic that is beyond the scope of this post, activation functions are applied via element-wise multiplication to the output of the neurons in a given layer of a neural network in order to allow the model to learn from non-linear data. Notably, the ReLU family of activation functions (e.g ReLU, Leaky ReLU, and Parametric ReLU) are commonly used in deep learning for this purpose, as well as for mitigating what is known as the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

Let's go over a simple example of how to perform element-wise multiplication with matrices $A$ and $B$.

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
tensor([[5, 12],[21, 32]])
```

### Dot Product Matrix Multiplication

With the easy operations out of the way, let's start diving into slightly more complex ones starting with _dot product multiplication_. The uses of dot production multiplication are almost too numerous to list out, but to highlight a few especially important ones:

-   **Gradient Descent**: Dot product multiplication is used to generate new prediction values after a single step of gradient descent by multiplying our input features (e.g $X$) by the calculated gradients associated with each feature respectively. It is also directly used in calculating the gradients for each feature, where we multiply the differences between our predicted values and actual $y$ values by what is known as the _transpose_ of our $X$ values. More on transposition in a bit!
-   **Neural Network Forward Pass**: During what is known as the _foward pass_ part of the training process in a neural network, dot product multiplication is used to compute the weighted sum for each neuron in a given layer by multiplying the input features (e.g $X$) values by the vector of the neuron's current weights. This weighted sum is what's then passed to an activation function via scalar multiplication, as we noted previously above.

Mathematically, dot production multiplication is accomplished by multiplying every element in each row of the first matrix by every element in each column of the second matrix, before adding each row's calculations together to produce a new matrix. Unlike most of our previous operations, dot product multiplication can not always be performed on matrices of equal size - the constraint is that the number of rows in the first matrix must equal the number of columns in the second matrix.

This might seem confusing, so let's go through an example with two matrices of identical rows and columns:

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

While a little more involved than element-wise multiplication, the resulting math is still not too bad, although we can imagine that the larger our matrices become, the more involved our calculations would be. Fortunately, PyTorch makes this sequence of operations incredibly easy so we don't have to actually worry about what's going on underneath the hood.

```python
# Matrix
A = torch.tensor([[1,2], [3,4]])
B = torch.tensor([[5,6], [7,8]])

torch.matmul(A, B)
```

```
tensor([[19, 22],[43, 50]])
```

Next, let's try dot product multiplication on two vectors, as is commonly done in the forward pass phase of training a neural network. Let's imagine that we have two vectors, $A$ and $B$, where $A$ represents the feature values in a hypothetical training set, while $B$ represents the current weights mapped to each feature in a given neuron:

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

Because the number of columns in vector $A$ equals the number of rows in vector $B$, we can proceed with dot product multiplication in order to calculate a weighted sum for the neuron:

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

Pretty cool, huh? If you'd like to practice yourself, [matrixmultiplication.xyz](http://matrixmultiplication.xyz/) is a great website for experimenting with performing dot product multiplication on matrices and vectors of your choosing! I highly recommend playing around with it to solidify your understanding of the underlying math, because visually it does a great job of making it easy to understand.

Before moving on, let's look at what happens if we try dot product multiplication on two matrices whose columns and rows don't align. A perfect example of this can be found in the gradient descent algorithm, which as a reminder, is expressed by the formula:

$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$

(_Note: If you need a refresher on how the gradient descent algorithm works, checkout [my last post](https://www.zacmessinger.com/posts/gradient-descent-scratch)_!)

Let's focus on the part of this algorithm where we need to calculate the total cost for each $α$ value by applying dot product multiplication on vector $D$ of the differences in our predicted and actual $y$ values with matrix $X$ of our input features, as represented by $(h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$.

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

Because vector $D$ has $1$ column while matrix $X$ has $4$ rows, we should not be able to perform dot production multiplication based on the constraints covered above. To verify this, let's try this out in PyTorch:

```python
y = torch.tensor([[-2], [-4], [-6], [-8]])
X = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4]])

y.matmul(X)
```

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1 and 4x2)
```

And voila, our assumption is correct! Believe or not, this is a common occurrence in machine and deep learning that frequently needs to be worked around, and luckily, there is a very straightforward solution. Let's move onto our next matrix operation, _transposition_, to find out how.

### Transposition

Matrix transposition is a mathematical operation that involves flipping a matrix over it's diagonal by swapping it's rows and columns, meaning the new resulting matrix will have the same number of columns as the initial matrix had rows. The easiest way to demonstrate this is visually, so let's go over a quick example using matrix $A$ as defined below:

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

This seems like it would be pretty helpful with the incompatible shape error we encountered above, where we could not perform dot product multiplication for vector $D$ and matrix $X$ because the number of columns in vector $D$ did not match the number of rows in matrix $A$. What if we transpose vector $D$, so that instead of having a 4x1 shape it now has a 1x4 shape?

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

Perfect! Now we can perform dot product multiplication to finish calculating the vector of total costs:

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

Just as we did with our previous matrix operations, let's now implement this programmatically. If we wanted to transpose the matrix by hand, the resulting function would look like the following:

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
print(f"A Dimensions: {torch.tensor(A).shape} | Transposed Dimensions: {torch.tensor(transposed).shape}")
```

```
A: [[-2], [-4], [-6], [-8]] | Transposed: [[-2, -4, -6, -8]]
A Dimensions: torch.Size([4, 1]) | Transposed Dimensions: torch.Size([1, 4])
```

Fortunately, we don't have to write out all this code because PyTorch makes it so easy:

```python
y = torch.tensor([[-2], [-4], [-6], [-8]])
X = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4]])


y.T.matmul(X)
```

```
tensor([[-20, -60]])
```

### Conclusion

If you've made it this far, thank you for taking the time to read my post! I hope I've demonstrated that the concepts of matrix addition and subtraction, scalar multiplication, element-wise multiplication, dot product multiplication, and transposition are not only fundamental to linear algebra but also play a pivotal role in the world of machine and deep learning.

I'll catch you next time!
