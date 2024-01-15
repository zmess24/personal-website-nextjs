---
title: "Gradient Descent from Scratch"
description: "Exploring the math behind GD..."
tags: ["Math"]
date: "1/17/2023"
link: "/posts/gradient-descent-scratch"
---

Learning machine learning can sometimes feel like venturing into a wondrous, yet complex rabbit hole of mathematical models and methods. One especially important algorithm that has seemed to pop up everywhere in my ML & DL studies is "Gradient Descent", and although I understood it's significance in helping ML models "learn" optimal parameter values, I didn't exactly understand how it worked - so that's what we'll be covering in this post! We'll code Linear Regression with Gradient Descent step-by-step, incrementally building this algorithm together.

If you've never heard of these terms, thats OK! Hopefully, this hands-on approach leads you to have the same "aha" moment that I had in grasping why Gradient Descent is such a useful tool in the ML ecosystem. While modern libraries like SciKit-Learn, PyTorch and TensorFlow provide handy built in methods for training models, if you're anything like me, the power in using those libraries correctly is to have a deep understanding of what the library methods are actually doing.

Brush off your high school linear algebra and calculus skillset, because here we go!

### Library Setup & Helper Functions

Before we go any further, let's import a few libraries and helper functions that we will use throughout the remainder of this post.

```python
# Library Imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import matplotlib as mpl
%matplotlib inline
```

```python
# Helper Functions
def create_scatter_plot(x, y, scale):
    mpl.style.use("seaborn-v0_8")
    plt.scatter(x=x, y=y)
    plt.axis(scale)
    # Set Labels
    plt.xlabel('carats of Gold')
    plt.ylabel('Price Sold')
    plt.title('carats of Gold vs Price Sold')


def create_cost_plot(X, y):
    mpl.style.use("seaborn-v0_8")
    plt.plot(X, y)
    # Set Labels
    plt.xlabel("Theta Values")
    plt.ylabel("Cost")
    plt.title("Theta vs Cost")
```

### Linear Regression Problem Setup

In order to understand the usefulness of Gradient Descent, let's start with a simple business premise. Imagine we are opening a jewelry price forecasting service that specializes in selling gold rings with a single ounce of silver, where the price is determined by the carats of gold in each ring plus the fixed price of silver. We can state our pricing methodology as $y = w(x) + c$, where:

-   $w$ is the price prediction
-   $x$ is the number of gold carats in the ring
-   $w$ is the price of a single carat of gold
-   $c$ is the fixed price of silver

In order to jump start the process of finding values for $w$ and $c$, we collect the following sales records from three seperate jewelry stores:

-   Store 1 sold a 1 carat gold ring for 150 dollars
-   Store 2 sold a 2 carat gold ring for 250 dollars
-   Store 3 sold a 3 carat gold ring for 250 dollars

This set can be expressed mathematically by what is known as a [system of equations](https://en.wikipedia.org/wiki/System_of_linear_equations). Our goal in solving the system is to determine the values of $w$ and $c$ that satisfy the conditions of all three equations.

$$
\begin{cases}
1(x) + c = 150\\
2(x) + c = 250\\
3(x) + c = 250\\
\end{cases}
$$

Immediately, we can see that we have contradictory information - there are no unique solutions for $w$ and $c$ that would satisfy the equations $2(x) + c = 250$ and $3(x) + c = 250$! Plotting these data points on a graph provides visual evidence of this conclusion, because as we can see, they do not form a straight line.

```python
create_scatter_plot(x=[1,2,3], y=[150, 250, 250], scale=[0,4,0,500])
```

![](/images/posts/linear-regression-scatch/figure-1.png)

In linear algebraic terms, this system would be classified as an incomplete, singular system, because there are no solutions for $w$ and $c$ that would solve all three equations. Therefore, in order for our jewelry forecasting service to output useful predictions for our customers, we will need to perform linear regression in order to find a line of best fit through our data.

### Defining a Linear Hypothesis

So where do we start? Well, if you recall from your linear algebra classes, a straight line can be defined by the equation $y = w(x) + c$. If this looks familiar, it's because it's the same equation as our initial pricing methodology from above! Let's quickly redefine what the parameters of the equation mean within the context of traditional linear algebra:

-   $y$ is the predicted output.
-   $w$ is the slope or gradient of the line.
-   $x$ is the input value.
-   $c$ is the bias term, or where the line should start on the y-intercept.

Given this revelation, it should come as no surprise that the linear regression hypothesis function we need to use to find optimal values for $w$ and $c$ is literally the same exact equation with slightly different notation.

$h(x) = θ₀ + θ₁x$

Where:

-   $h(x)$ is the predicted output for a provided input value of $x$.
-   $θ₀$ or _theta0_, is the bias term, or where the line should start on the y-intercept.
-   $θ₁$ or _theta1_, is the slope or gradient of the line.
-   $x$ is the input value.

And therein lies the mathmatical goal of linear regression - to find optimal values for $θ₀$ and $θ₁$ so that we can create a line of best fit for our dataset. To do that, we will use an optimization algorithm called _Gradient Descent_ in combination with a cost function called the _Mean Squared Error_.

_Before moving forward, it's worth noting that there are a couple ways to solve this linear regression hypothesis, but gradient descent is the most common in the ML / AI space due to its computational efficiency on large datasets. However, it is possible to solve for $θ₀$ and $θ₁$ using straight linear algebra, but that topic is a little bit beyond the scope of this post. Perhaps I'll cover how to do that in a future one, because the math is fun to work through!_

### Defining a Cost Function

Now that we're ready to find values for $θ₀$ and $θ₁$, the first question we need to ask ourselves is how can we get a baseline measurement for how well any future values we assign to them fit our data? Fortunately, there is a relatively straight forward statistical method to accomplish this called the _Mean Squared Error_. Let's lay it out formulaically before diving into it:

$
MSE = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)^2
$

Where:

-   $n$ is the number of training examples.
-   $x_i$ represents the independent variables or features for the $i^{th}$ training example.
-   $y_i$ is the actual value of the dependent variable for the $i^{th}$ training example.
-   $h_\theta(x_i)$ is the linear hypothesis function, which is the equation we had defined above $h(x) = θ₀ + θ₁x$

This may look intimidating, but as I mentioned, it's simpler than it looks. The main objective of the Mean Squared Error is to get the average difference or "cost" between our predicted $y$ values (as determined by our theta values), and the actual $y$ values in our data. In theory, **the closer the cost is to zero, the better our theta values fit our data**. How we do know this? Because a line that perfectly fits our data would have no difference between the predicted values and the observed values, meaning it would be a complete, non-singular system with a cost of $0$.

The MSE method works by performing the following steps:

1. Calculate the differences between the predicted $y$ values and observed $y$ values.
2. Square each difference to ensure a positive number is produced (the cost can't be negative).
3. Add the squared differences together to produce a total cost.
4. Multiply the sum by $\frac{1}{2n}$, where _n_ represents the total samples in the training set. In case you're wondering, we multiply by $2n$ instead of $n$ as a way to offset the squaring of the differences in the first step.

I know this may still seem confusing, so to make this easier to understand, let's put aside our jewelry store dataset for now and look at an even more basic one with the below observations:

$$
\begin{cases}
1(x) = 3\\
2(x) = 6\\
3(x) = 9\\
\end{cases}
$$

Pretty easily, we can see that solving this system of equations would result in $x = 3$ with a linear hypothesis function of $h(x) = 0 + 3x$. But, let's assume solving this system isn't quite that straight forward and so we decide to intitialize our linear hypothesis function with a $θ₁$ value of $0$. Let's now calculate the MSE of $θ₁$ step by step.

First, let's get our predicted $y$ values by plugging in $0$ into our linear hypothesis function:

$$
\begin{cases}
h(1) = 0 + 0(1) = 0\\
h(2) = 0 + 0(2) = 0\\
h(3) = 0 + 0(3) = 0
\end{cases}
$$

Calculate the differences between the predicted $y$ values and observed $y$ values.

$$
h_\theta(x_i) - y_i = {\begin{pmatrix} 0-3 & 0-6 & 0-9 \end{pmatrix}}\\
h_\theta(x_i) - y_i = {\begin{pmatrix} -3 & -6 & -9 \end{pmatrix}}
$$

Square each difference to ensure a positive number is produced (the cost can't be negative).

$$
(h_\theta(x_i) - y_i)^2 = {\begin{pmatrix} -3^2 & -6^2 & -9^2 \end{pmatrix}}\\
(h_\theta(x_i) - y_i)^2 = {\begin{pmatrix} 9 & 36 & 81 \end{pmatrix}}
$$

Add the squared differences together to produce a total cost.

$$
\sum_{i=1}^{n} (h_\theta(x_i) - y_i)^2 = 9 + 36 + 81 = 126
$$

Multiply the sum by $\frac{1}{2n}$, where _n_ represents the total samples in the training set

$$
MSE = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)^2 = \frac{1}{2(3)}\ * 126 = 21\\
$$

Fantastic! We've calculated an initial cost of $21$ using the Mean Squared Error method, and now we have a baseline to work with as we attempt to find an optimal values for $θ₀$ and $θ₁$. Remember, our goal is to decrease this cost as close to $0$ as possible with any future values we assign to them. We'll go over how to do so in the next section using Gradient Descent.

### Calculating a Gradient

With a cost function now in place, we're ready to figure out how to tune our $θ₀$ and $θ₁$ values in order to bring down the total cost of our pricing hypothesis function. We will do so by calculating what is called the "gradient" (mathamtically known as the partial derivative) of our cost function, i.e measuring how small changes in our theta values affect the overall cost of our predictions. This concept of gradients is crucial, so let's go over it in more depth before moving into the actual Gradient Descent formula.

To help us, let's go through an imaginary exercise. Imagine you're a hiker standing on a large hill that has suddenly become incredibly foggy, so you can only see the ground directly in front of you. Due to the changing weather conditions, you decide you need to descend the hill as quickly as possile. Logically speaking, the easiest way to accomplish this would be to look around you in every direction, take a step where the path descends the steepest, and repeat this process until you reach the bottom. The "gradient", in this context, would be your sense of whether the ground underneath your feet slopes up or down with each step you take. In the context of Linear Regression, a gradient is largly the same - it mathmatically guides us in knowing the direction and extent to which we should adjust our theta values to reduce the cost of our predictions.

Let's zoom out for a second by looking at the MSE's for a range of potential $θ₁$ values for our linear hypothesis function.

![](/images/posts/linear-regression-scatch/figure-10.png)

Notice something interesting? As $θ₁$ approaches 3 (which we know to be the best answer), our cost gradually descends until it reaches $0$, and then once $θ₁$ moves past $3$, the cost gradually increases at exactly the same rate. Plotting this out on a graph results in a parabola:

```python
X=[0, 1, 2, 3, 4, 5, 6]
y=[21.0, 9.33, 2.33, 0, 2.33, 9.33, 21.0]

create_cost_plot(X, y)
```

![](/images/posts/linear-regression-scatch/figure-2.png)

Visually, we can see that in order to find optimal values for $θ₁$, we simply have to navigate towards the bottom of the parabola relative to our current location (just like in our thought exercise from above). The question is, how can we leverage the gradient to do so? Let's walk through it.

The equation for calculting the gradient should look familar to us, since it's more or less the same as the Mean Squared Error method we used above. Notice this time we are not squaring the difference between our predicted $y$ values and observed $y$ values, because the sign of the gradient tells us whether the slope is ascending or descending (more on this in a bit).

$
\frac{\partial}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i\\
$

Let's find the gradient of our current $θ₁$ value of $0$ (which had an MSE cost of $21$) by first generating our predictions.

$$
\begin{cases}
h(1) = 0 + 0(1) = 0\\
h(2) = 0 + 0(2) = 0\\
h(3) = 0 + 0(3) = 0
\end{cases}\\
$$

And now the notation:

$$
(h_\theta(x_i) - y_i) \cdot x_i = (0−3)⋅1+(0−6)⋅2+(0−9)⋅3\\
\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i = \frac{1}{3}\ (-42)\\
\frac{\partial}{\partial \theta_1} = -14
$$

### Applying a Learning Rate

With a gradient of $-14$ now calculated for $θ₁$, all that remains is to determine how we can use it to make a decision on whether to increase or decrease $θ₁$, and by how much.

Luckily, the answer is incredibly intuitive. If the gradient is negative (i.e the cost decreases with respect to $θ₁$), we know we need to increase $θ₁$, and if the gradient is positive (i.e the cost increases with respect to $θ₁$), we know we need to decrease $θ₁$. Mathematically, we can accomplish this by simply subtracting the gradient from the current theta value, since subtracting a negative number from a positive number is the same as performing addition, while subtracting a positive number from a positive number is straight subtraction!

Subtracting $-14$ from $θ₁$ results in a new theta value of $14$ ($0 + 14$), which if we ran MSE on, would result in a new cost of $282.33$ compared to $21$ previously.

```python
X=[0, 1, 2, 3, 4, 5, 6, 14]
y=[21.0, 9.33, 2.33, 0, 2.33, 9.33, 21.0, 282.33]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0,14], y=[21, 282.33], color="r")
plt.plot([0, 14], [21, 282.33], color="r")
```

![](/images/posts/linear-regression-scatch/figure-3.png)

Uh Oh. Although we increased $θ₁$ in the correct direction, we actually increased it by far too much since our cost dramatically went up! In order to avoid this, we can multiply the gradient by what's called a _Learning Rate_ ($\alpha$) in order to decrease the size of the steps we take towards the vertex of the parabola. Learning Rates are generally very small, with most Gradient Descent algorithms defaulting to $0.01$.

Let's change the size of our step by applying a learning rate of $0.01$ and recalculate the MSE:

$
θ₁ = 0 + 0.01(14) = 0.14\\
MSE = 19.09
$

```python
X=[0, 0.14, 1, 2, 3, 4, 5, 6]
y=[21.0, 19.09, 9.33, 2.33, 0, 2.33, 9.33, 21.0]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0, 0.14], y=[21, 19.09], color="r")
plt.plot([0, 0.14], [21, 19.09], color="r")
```

![](/images/posts/linear-regression-scatch/figure-4.png)

Much better! We've decreased our cost from $21$ to $19.09$ by increaasing $θ₁$ to $0.14$.

### Gradient Descent

With that, we've actually just run a single iteration of Gradient Descent! Here is the full formula in its entirety.

$\theta_j := \theta_j - \alpha \frac{1}{m} \sum*{i=1}^{m} (h\_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$

The trick to Gradient Descent is to keep re-running it either until a set amount of iterations is reached (which gives it the time it needs to find optimal theta values), or we reach convergence (i.e we've reached the vertex of the cost parabola). The amazing thing about this optimization algorithm is that it will also update all of our potential theta values concurrently, meaning it scales incredibly well with complex datasets. Let's run it one more time using our new $θ₁$ value of $0.14$ to see the power of iteration in action.

First, let's update our predictions.

$$
\begin{cases}
h(1) = 0 + 0.14(1) = 0.14\\
h(2) = 0 + 0.14(2) = 0.28\\
h(3) = 0 + 0.14(3) = 0.42
\end{cases}\\
$$

And now, let's re-run the formula.

$$
(h_\theta(x_i) - y_i) \cdot x_i = (0.14 − 3)⋅1+(0.28 − 6)⋅2+(0.42 − 9)⋅3\\
\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i = \frac{1}{3}\ (-40.04)\\
\frac{\partial}{\partial \theta_1} = -13.3467\\
θ₁ = 0.14 + 0.01(13.3467) = 0.273467\\
MSE = 17.35
$$

```python
X=[0, 0.14, 0.273467, 1, 2, 3, 4, 5, 6]
y=[21.0, 19.09, 17.35, 9.33, 2.33, 0, 2.33, 9.33, 21.0]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0, 0.14, 0.273467], y=[21, 19.09, 17.35], color="r")
plt.plot([0, 0.14, 0.273467], [21, 19.09, 17.35], color="r")
```

![](/images/posts/linear-regression-scatch/figure-5.png)

Now that we have a firm understanding of how gradient descent works mathematically in the context of linear regression, let's apply it on a more robust dataset by circling back to the jewelry price forecasting service we had originally set out to solve.

### Putting it all Together

To make this more interesting, let's scale up our jewelry store dataset by randomly generating 100 data points where each ring has between 0 and 25 carats of gold, and the price per carat of gold ranges from 40 to 60 dollars. We'll assume each ring has a single ounce of silver worth $30.

```python
# Create random 'X' dataset between 0 and 25 carats of gold
np.random.seed(42)
X = 25 * np.random.rand(100,1)

# Create random 'y' dataset that multiplys each carat of gold by $40 - $60
y = np.random.rand(100,1)

for i in range(0, len(y)):
    random_gold_price = random.randrange(40, 60)
    silver_price = 30
    price = silver_price + (random_gold_price * X[i][0])
    y[i][0] = round(price, 2)

# Convert numpy arrays to pandas dataframes for easy viewing
data = np.concatenate((X, y), axis=1)
df = pd.DataFrame(data, columns=['carats_of_gold', 'price_sold'])
```

We can visualize the data across a scatterplot and clearly see that a straight line can not pass through every point on the plot, and therefore, we will need to perform regression to find a line of best.

```python
create_scatter_plot(x=df["carats_of_gold"], y=df["price_sold"], scale=[0,25,0,1500])
```

![](/images/posts/linear-regression-scatch/figure-6.png)

If we zoom in on this graph a little closer, we can see the y intercept (i.e our the price of our silver) seems to start right around where we'd expect, between 30 & 40 dollars given the variable price of gold.

```python
create_scatter_plot(x=df["carats_of_gold"], y=df["price_sold"], scale=[0,1.5,0,100])
```

![](/images/posts/linear-regression-scatch/figure-7.png)

With a robust set data now curated, let's write out our Mean Squared Error and Gradient Descent functions.

```python
# 𝐽(𝜃)=1/2𝑛 ∑𝑛𝑖=1(ℎ𝜃(𝑥𝑖)−𝑦𝑖)2
def mean_squared_error(X, y, theta):
    m = y.shape[0]
    h = X.dot(theta)
    J = (1/(2*m)) * (np.sum((h - y)**2))
    return J

# 𝜃𝑗 = 𝜃𝑗 − 𝛼 * 1𝑚 ∑ 𝑚𝑖=1(ℎ𝜃(𝑥(𝑖))−𝑦(𝑖))⋅𝑥(𝑖)
def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    cost_history = []

    for i in range(0, num_iters):
        predictions = X.dot(theta)
        difference = predictions - y
        gradient = (1/m) * (difference.T.dot(X))
        theta = theta - (alpha * gradient.T)
        cost = mean_squared_error(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history
```

And finally, let's run Gradient Descent after a little data preprocessing to get our theta values!

```python
m = df.shape[0]
X = np.hstack((np.ones((m,1)), df.carats_of_gold.values.reshape(-1,1)))
y = np.array(df.price_sold.values).reshape(-1,1)
theta = np.zeros(shape=(X.shape[1],1))
iterations = 1000
alpha = 0.01

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print(f"Silver Price: {theta[0][0]}")
print(f"Gold Price: {theta[1][0]}")
print(f"Average Prediction Error: ${np.sqrt(cost_history[999])}")
```

```
Silver Price: $28.61838193739571
Gold Price: $50.36120667705771
Average Prediction Error: $58.15829026058178
```

Not too bad! After running a $1000$ iterations of Gradient Descent, our Linear Regression model arrived at a silver price of $28.62$, and a gold price of $50.36$. If we overlay these theta values as a line ($y= 50.36(x) + 28.62$) across the scatterplot of our observations, we can indeed see it matches our data!

```python
# Create Plot
create_scatter_plot(x=df["carats_of_gold"], y=df["price_sold"], scale=[0,25,0,1500])

# Plot line of best fit.
plt.plot(X[:,1], X.dot(theta), color='r')
```

![](/images/posts/linear-regression-scatch/figure-9.png)

It can also be helpful to look at a graph comparing the drop in the overall cost of our model against the number of iterations that we ran. In our case, we can see that Gradient Descent reached convergence relatively quickly.

```python
iterations = np.arange(0, 1000)
plt.scatter(x=iterations, y=cost_history)

# Set Labels
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History vs Gradient Descent Iterations")
```

![](/images/posts/linear-regression-scatch/figure-8.png)

Finally, let's test our new jewelry pricing model by using it to predict the price of a ring with 15 carats of gold.

```python
datapoint = [1, 15]
prediction = np.dot(datapoint, theta)

print(f"Price Prediction: ${prediction[0]}")
```

```
Price Prediction: $784.0364820932614
```

### Conclusion

While this was a relatively basic implementation of Linear Regression, the core concept of using Gradient Descent to find optimal coefficient values for a function's parameters applies similarly in more complex datasets. Notably, Gradient Descent plays a crucial role in training neural networks by optimizing their weights during the backwards propagation process. It's also worth noting that there are variations of the Gradient Descent algorithm, such as Batch, Stochastic, and Mini-batch Gradient Descent, each offering advantages for different types of datasets. These variations are something I look forward to exploring in future posts. I hope you found this post insightful and enjoyable, just as I did in creating it! Stay tuned for more, and happy learning!
