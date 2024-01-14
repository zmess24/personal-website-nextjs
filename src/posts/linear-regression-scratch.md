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
# Create Plot
mpl.style.use("seaborn-v0_8")
plt.scatter(x=[1,2,3], y=[150, 250, 250])
plt.axis([0,4,0,500])

# Set Lables
plt.xlabel('Karots of Gold')
plt.ylabel('Price Sold')
plt.title('Karots of Gold vs Price Sold')
```
