---
title: "Linear Regression from Scratch"
description: "Prompt engineering vs fine-tuning..."
tags: ["Math"]
date: "1/17/2023"
link: "/posts/linear-regression-scratch"
---

As I've started to venture deeper and deeper down the ML & DL rabbit hole, Gradient Descent has been a topic that has come up frequently. At a high level, Gradient Descent is a useful training method used by regression algorithms (and most notably neural networks) to find optimal values for a given function parameters such that the loss of the cost function becomes globally minimized.

Don't know what any of that means? That's OK! The goal of this post is to deconstruct what Gradient Descent is within the context of Linear Regression into layman's terms to provide a better understanding of why it is such an important tool in the ML & DL toolkit. While many modern ML & DL libraries (e.g SciKit-Learn, Tensorflow, Pytorch) abstract away the need to understand what's going on underneath the hood when training a model, if you're anything like me, the power in using those libraries correctly is to have a deep understanding of what the library methods are actually doing. In other words, understand the math!

And with that, brush off your high school linear algebra and calculus skillset! Here we go.

### Library Setup & Helper Functions

Before we go any further, let's import a few libraries that we will use throughout the remainder of this post.

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
    plt.xlabel('Karots of Gold')
    plt.ylabel('Price Sold')
    plt.title('Karots of Gold vs Price Sold')


def create_cost_plot(X, y):
    mpl.style.use("seaborn-v0_8")
    plt.plot(X, y)
    # Set Labels
    plt.xlabel("Theta Values")
    plt.ylabel("Cost")
    plt.title("Theta vs Cost")
```

### Linear Regression Problem Setup

Let's start with a simple premise. Imagine we are opening a jewelry price prediction company that specializes in selling gold rings with a specific amount of silver, where the price is determined by the carats of gold in each ring plus the fixed price of the silver. We can express our pricing hypothesis as $y = w(x) + c$, where:

-   $w$ is our prediction
-   $x$ is the number of gold carats in a the ring
-   $w$ is the price of a single carat of gold
-   $c$ is the fixed price of silver

In order to jump start the process of finding values for $w$ and $c$, we collect the following data from 3 different jewelry stores:

-   One store sold a single carat gold ring for 150 dollars
-   Another sold a 2 carat gold ring for 250 dollars
-   The last sold a 3 carat gold ring for 250 dollars

We can express these datapoints as a <a href="https://en.wikipedia.org/wiki/System_of_linear_equations">system of equations</a>, with the intent being to solve the system in order to find values for $w$ and $c$ that would satisfy all three equations:

$$
\begin{cases}
1(x) + c = 150\\
2(x) + c = 250\\
3(x) + c = 250\\
\end{cases}
$$

Immediately, we can see that we have contradictory information - there are no unique solutions for $a$ and $b$ that would satisfy the equations $2(x) + c = 250$ and $3(x) + c = 250$, meaning this is an incomplete, singular system. Plotting these data points across an X & Y plot provides visual evidence of this conclusion, because as we can see, these three points do not form a straight line:

```python
create_scatter_plot(x=[1,2,3], y=[150, 250, 250], scale=[0,4,0,500])
```

![](/images/posts/linear-regression-scatch/figure-1.png)

Since we don't have a unique solution for $w$ and $c$, we now know we will need to perform linear regression in order to find a line of best fit so that our pricing algorithm can make accurate predictions. But how can we do this? Well, if you recall from your linear algebra classes, a straight line can be defined by the equation $y = w(x) + c$, which should look familiar as it is exactly the same equation as the pricing hypothesis we had previously defined! Let's quickly redefine what the equation's parameters mean with the context of traditional linear algebra:

-   $y$ is the predicted output.
-   $w$ is the slope or gradient of the line.
-   $x$ is the input value.
-   $c$ is the bias term, or where the line should start on the y-intercept.

Given this revelation, it should come as no surprise that the linear regression hypothesis equation we will use to find optimal values for $w$ and $c$ follows this same exact formula:

$h(x) = θ₀ + θ₁x$

-   $θ₀$ or _theta 0_, is the bias term, or where the line should start on the y-intercept.
-   $θ₁$ or _theta 1_, is the slope or gradient of the line.
-   $x$ is the input value.

And therein lies the goal of linear regression - to find optimal values for _theta0_ ($θ₀$) and _theta1_ ($θ₁$) so that we can create a line of best fit for our dataset. To do that, we will use an optimization algorithm called _Gradient Descent_ in combination with a cost function called the _Mean Squared Error_.

Before moving forward, it's worth noting that there are a couple ways to solve this linear regression hypothesis, but gradient descent is common in the ML / AI space mainly due to its computational efficiency on large datasets. However, it is possible to solve for $θ₀$ and $θ₁$ using straight linear algebra via the <a href="https://en.wikipedia.org/wiki/Linear_least_squares#Example">Least Squares method</a>, but that topic is a little bit beyond the scope of this post. Perhaps I'll cover how to do that in a future one, because the formula is fun to work through!

### Mean Squared Error

Now that we're ready to find values for _theta0_ ($θ₀$) and _theta1_ ($θ₁$), the first question we need to ask ourselves is how can we get a baseline measurement for how well any future values we assign to them fit our data? Fortunately, there is a relatively straight forward formula to accomplish this called the _Mean Squared Error_. Let's lay it out in all it's glory before diving into it:

$J(\theta) = \frac{1}{2n} \sum*{i=1}^{n} (h*\theta(x_i) - y_i)^2$

Where:

-   $J(\theta)$ represents the cost of the current theta values.
-   $n$ is the number of training examples.
-   $x_i$ represents the independent variables or features for the $i^{th}$ training example.
-   $y_i$ is the actual value of the dependent variable for the $i^{th}$ training example.
-   $h_\theta(x_i)$ is the hypothesis function, which is the equation we had defined above $h(x) = θ₀ + θ₁x$

This may look intimidating, but as I mentioned, it's actually simpler than it looks. The main objective of the Mean Squared Error is to get the average difference or "cost" between our predicted $y$ values (as determined by our theta values), and the actual $y$ values in our data. In theory, the closer the cost is to zero, the better we know our theta values fit our data. How we do know this? Because a line that perfectly fits our data would have no difference between the predicted values and the observed values, meaning it would be a complete, non-singular system with a cost of $0$.

The Mean Squared Error works by performing the following steps:

1. Calculate the differences between our predicted $y$ values, and our observed $y$ values.
2. Square each difference to ensure we produce a positive number (after all, a cost can't be negative).
3. Add the squared differences together to produce a total cost.
4. Multiply the sum by $\frac{1}{2n}$, where _n_ represents the total samples in our training set (_we multiply by \_2n_ instead of _n_ as a way to offset the squaring of the differences in the first step.\_)

If you're still confused, that's OK! Let's go over this mathematically using the data provided by our jewelry stores above. To start, let's initialize our our hypothesis function ($h(x) = θ₀ + θ₁x$) with $θ₀$ and $θ₁$ values of $0$ and $0$, meaning that our initial cost predictions for the gold rings in our dataset are as follows:

$$
\begin{cases}
h(1) = 0 + 0(1) = 0\\
h(2) = 0 + 0(2) = 0\\
h(3) = 0 + 0(3) = 0
\end{cases}
$$

Next, let's calculate the difference between our predicted $y$ values and our observed $y$ values:

$$
(h_\theta(x_i) - y_i) = {\begin{pmatrix} 0-150 & 0-250 & 0-250 \end{pmatrix}}
$$

Then, let's square the differences before adding them together:

$$
\sum_{i=1}^{n}(h_\theta(x_i) - y_i)^2 = -150^2 + -250^2 + -250^2 = 147,500\\
$$

Finally, let's multiply the sum by $\frac{1}{2n}$:

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x_i) - y_i)^2 = \frac{1}{2(3)}\ * 147,500 =  24,583.33\\
$$

Fantastic! We've calculated our initial cost using the Mean Squared Error method, and now we have a baseline to work with as we attempt to find optimal theta values in order to find a line of best fit. With that, we're almost at the finish line. We're ready for _Gradient Descent_.

### Calculating a Gradient

Gradient Descent is an optimization algorithm widely used in machine learning to methodically reduce the cost of a model's predictions, which as discussed, we need to get as close to zero as possible. This process works by calculating the "gradient" (mathematically known as the partial derivative) of our cost function, i.e measuring how small changes in our theta values will affect the overall cost of our predictions. Put more plainly, Gradient Descent guides us in knowing the direction and extent to which we should adjust our theta values to enhance the accuracy of our predictions. This concept of gradients is crucial, so let's go over why it's so important before discussing the actual formula.

To make this easier to understand, let's put aside our jewelry store dataset for now and look at an even more basic one with the below observations:

$$
\begin{cases}
1(x) = 3\\
2(x) = 6\\
3(x) = 9\\
\end{cases}
$$

Pretty easily, we can see that solving this system of equations would result in $x = 3$ with a linear hypothesis function of $h(x) = 0 + 3x$. However, let's assume solving this system isn't quite that straightforward. Why don't we use the Mean Squared Error method we previously defined to map out the costs associated with a few potential θ₁ values?

| θ₁   | 0    | 1    | 2    | 3    | 4    | 5    | 6    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Cost | 42.0 | 18.7 | 4.67 | 0.00 | 4.67 | 18.7 | 42.0 |

Notice something interesting? As $θ₁$ approaches 3 (which we know to be the best answer), our cost gradually descends until it reaches $0$, and then once $θ₁$ moves past $3$, the cost gradually increases at exactly the same rate. Plotting this out on a graph results in a parabola:

```python
X=[0, 1, 2, 3, 4, 5, 6]
y=[42.0, 18.7, 4.7, 0, 4.7, 18.7, 42.0]

create_cost_plot(X, y)
```

![](/images/posts/linear-regression-scatch/figure-2.png)

Notice something interesting? As $θ₁$ approaches 3 (which we know to be the best answer), our cost gradually descends until it reaches $0$, and then once $θ₁$ moves past $3$, the cost gradually increases at exactly the same rate. Plotting this out on a graph results in a parabola:

Visually, we can see that in order to find optimal values for $θ₁$, we simply have to navigate towards the bottom of the parabola relative to where we are now, since the vertex represents a cost of $0$. As an example, if $θ₁$ was initialized with a value of $0$, we as humans would intuitively know we would need to increase $θ₁$ in order to get closer to the vertex. The question is - how can we go through that thought process mathematically? That's where calculating the gradient comes in! By looking at the slope of the parabola relative to where we currently are, we can determine which direction to move in.

The formula for calculating the gradient should a little familiar to us, since it's largely the same as the Mean Squared Error method we used above:

$
\frac{\partial}{\partial \theta_1} \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i\\
$

Let's try it on our dataset by initializing our hypothesis function with $θ₁$ and $θ₀$ values of $0$:

$$
\begin{cases}
h(1) = 0 + 0(1) = 0\\
h(2) = 0 + 0(2) = 0\\
h(3) = 0 + 0(3) = 0
\end{cases}\\
$$

And now the notation:

$$
h_\theta(x_i) - y_i) \cdot x_i = (0−3)⋅1+(0−6)⋅2+(0−9)⋅3\\
\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i = \frac{1}{3}\ (-42)\\
\frac{\partial}{\partial \theta_1} \text{MSE} = -14
$$

### Gradient Descent

With a gradient of $-14$ now calculated for $θ₁$, the last questions we need to answer are how can we use it to help determine in which direction to move $θ₁$, and by how much? Luckily, the answer is incredibly intuitive - if the gradient is negative (i.e the cost decreases with respect to $θ₁$), we know we need to increase $θ₁$, and if the gradient is positive (i.e the cost increases with respect to $θ₁$), we know we need to decrease $θ₁$. Mathematically, we can accomplish this by simply subtracting the gradient from the current theta value, since subtracting a negative number from a positive number is the same as performing addition, while subtracting a positive number from a positive number is straight subtraction!

Subtracting $-14$ from our $θ₁$ results in a new theta value of $14$, which if we ran Mean Squared Error on, would result in a new cost of $1302$ compared to $42$ previously.

```python
X=[0, 1, 2, 3, 4, 5, 6, 14]
y=[42.0, 18.7, 4.7, 0, 4.7, 18.7, 42.0, 1302.0]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0,14], y=[42, 1302.0], color="r")
plt.plot([0, 14], [42, 1302.0], color="r")
```

![](/images/posts/linear-regression-scatch/figure-3.png)

Uh Oh - although we increased $θ₁$ in the correct direction, we increased it by far too much since our cost actually went up! In order to avoid this, we can multiply our gradient with what's called a Learning Rate $\alpha$ in order to decrease the size of the "steps" we take towards the vertex of the parabola. Learning Rates are generally very small, and most Gradient Descent algorithms default to $0.01$.

Let's change the size of our step by applying a learning rate of $0.01$ and rerun the MSE:

$
θ₁ = 0 + 0.01(14) = 0.14\\
J(\theta) = 38.17
$

```python
X=[0, 0.14, 1, 2, 3, 4, 5, 6]
y=[42.0, 38.17, 18.7, 4.7, 0, 4.7, 18.7, 42.0]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0, 0.14], y=[42, 38.17], color="r")
plt.plot([0, 0.14], [42, 38.17], color="r")
```

![](/images/posts/linear-regression-scatch/figure-4.png)

Much better! We've decreased our cost from 42 to 38.17 by making a small increase in $θ₁$. And with that, we've actually just run a single iteration of Gradient Descent! Here is the full formula in its entirety:

$\theta_j := \theta_j - \alpha \frac{1}{m} \sum*{i=1}^{m} (h\_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$

The trick to Gradient Descent is to keep re-running the formula either until we reach a set amount of iterations (which gives it the time it needs to find optimal theta values), or until our cost levels out (which means we've reached the bottom of the cost parabola). Let's run one more iteration to see that magic in action!

First, let's calculate our new predictions using the updated value of $θ₁$:

$$
\begin{cases}
h(1) = 0 + 0.14(1) = 0.14\\
h(2) = 0 + 0.14(2) = 0.28\\
h(3) = 0 + 0.14(3) = 0.42
\end{cases}\\
$$

Then, let's re-run the formula:

$$
h_\theta(x_i) - y_i) \cdot x_i = (0.14 − 3)⋅1+(0.28 − 6)⋅2+(0.42 − 9)⋅3\\
\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i) \cdot x_i = \frac{1}{3}\ (-40.04)\\
\frac{\partial}{\partial \theta_1} \text{MSE} = -13.3467\\
θ₁ = 0.14 + 0.01(13.3467) = 0.273467\\
J(\theta) = 34.69
$$

Amazing! We've gotten our cost down even closer to 0!

```python
X=[0, 0.14, 0.273467, 1, 2, 3, 4, 5, 6]
y=[42.0, 38.17, 34.69, 18.7, 4.7, 0, 4.7, 18.7, 42.0]

create_cost_plot(X=X, y=y)
plt.scatter(x=[0, 0.14, 0.273467], y=[42, 38.17, 34.69], color="r")
plt.plot([0, 0.14, 0.273467],[42, 38.17, 34.69], color="r")
```

![](/images/posts/linear-regression-scatch/figure-5.png)

Now that we have a firm understanding of how gradient descent works mathematically in the context of linear regression, let's apply it on a more robust dataset by circling back to the jewelry pricing hypothesis we had originally set out to solve.

### Putting it all Together

To make this more interesting, let's scale up our jewelry store dataset by randomly generating 100 data points where each ring has between 0 and 25 carats of gold, and the price per carat of gold ranges from 40 to 60 dollars. We'll assume each ring has a constant of 30 dollars worth of silver.

```python
# Create random 'X' dataset between 0 and 25 carots of gold
np.random.seed(42)
X = 25 * np.random.rand(100,1)

# Create random 'y' dataset that multiplys each carot of gold by $40 - $60
y = np.random.rand(100,1)

for i in range(0, len(y)):
    random_gold_price = random.randrange(40, 60)
    silver_price = 30
    price = silver_price + (random_gold_price * X[i][0])
    y[i][0] = round(price, 2)

# Convert numpy arrays to pandas dataframes for easy viewing
data = np.concatenate((X, y), axis=1)
df = pd.DataFrame(data, columns=['carots_of_gold', 'price_sold'])
```

We can visualize the data across a scatterplot and clearly see that a straight line can not pass through every point on the plot, and therefore, we will need to perform regression to find a line of best.

```python
create_scatter_plot(x=df["carots_of_gold"], y=df["price_sold"], scale=[0,25,0,1500])
```

![](/images/posts/linear-regression-scatch/figure-6.png)

If we zoom in on this graph a little closer, we can see the y intercept (i.e our the price of our silver) seems to start right around where we'd expect, between 30 & 40 dollars given the variable price of gold.

```python
create_scatter_plot(x=df["carots_of_gold"], y=df["price_sold"], scale=[0,1.5,0,100])
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
X = np.hstack((np.ones((m,1)), df.carots_of_gold.values.reshape(-1,1)))
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

```python
iterations = np.arange(0, 1000)
plt.scatter(x=iterations, y=cost_history)

# Set Labels
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History vs Gradient Descent Iterations")
```

![](/images/posts/linear-regression-scatch/figure-8.png)

```python
# Create Plot
create_scatter_plot(x=df["carots_of_gold"], y=df["price_sold"], scale=[0,25,0,1500])

# Plot line of best fit.
plt.plot(X[:,1], X.dot(theta), color='r')
```

![](/images/posts/linear-regression-scatch/figure-9.png)

Let's now test our new pricing model by using it to predict the price of a ring with 15 karots of gold.

$$
h(15) = 28.62 + 50.36(15)\\
h(15) = 784.02
$$

```python
datapoint = [1, 15]
prediction = np.dot(datapoint, theta)

print(f"Price Prediction: ${prediction[0]}")
```

```
Price Prediction: $784.0364820932614
```

### Conclusion
