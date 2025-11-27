#!/usr/bin/env python
# coding: utf-8

# # Ungraded Lab: The Hello World of Deep Learning with Neural Networks

# Like every first app, you should start with something super simple that shows the overall scaffolding for how your code works. In the case of creating neural networks, one simple case is where it learns the relationship between two numbers. So, for example, if you were writing code for a function like this, you already know the 'rules': 
# 
# 
# ```
# def hw_function(x):
#     y = (2 * x) - 1
#     return y
# ```
# 
# So how would you train a neural network to do the equivalent task? By using data! By feeding it with a set of x's and y's, it should be able to figure out the relationship between them. 
# 
# This is obviously a very different paradigm from what you might be used to. So let's step through it piece by piece.
# 

# ## Imports
# 
# Start with the imports. Here, you are importing [TensorFlow](https://www.tensorflow.org/) and calling it `tf` for convention and ease of use.
# The framework you will use to build a neural network as a sequence of layers is called [`keras`](https://keras.io/) and is contained within tensorflow, so you can access it using `tf.keras`.
# 
# You then import a library called [`numpy`](https://numpy.org) which helps to represent data as arrays easily and to optimize numerical operations.

# In[2]:


import tensorflow as tf
import numpy as np


# ## Generating the Data
# 
# Before creating a neural network you will generate the data that you will be working with. In this case, you are taking 6 X's and 6 Y's. You can see that the relationship between these is `y=2x-1`, so where `x=-1`, `y=-3` etc. 
# 
# The standard way of declaring model inputs and outputs is to use `numpy`, a Python library that provides lots of array type data structures. You can specify these values by building numpy arrays with [`np.array()`](https://numpy.org/doc/stable/reference/generated/numpy.array.html). Later in this course, you will learn how to use built-in `TensorFlow` functions to work with inputs.

# In[3]:


# Declare model inputs and outputs for training
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# ## Define and Compile the Neural Network
# 
# Next, you will create the simplest possible neural network. It has 1 layer with 1 neuron. You will build this model using Keras' [Sequential](https://keras.io/api/models/sequential/) class which allows you to define the network as a sequence of [layers](https://keras.io/api/layers/). You can use a single [Dense](https://keras.io/api/layers/core_layers/dense/) layer to build this simple network.
# 
# It is good practice to define the expected shape of the input to the model. In this case, you see that each element in `xs` is a scalar, which you can also treat as a 1-dimensional vector. You can define this shape through the [tf.keras.Input()](https://www.tensorflow.org/api_docs/python/tf/keras/Input) object and its `shape` parameter as shown below. Take note that this is not a layer so even if there are two lines of code inside `Sequential` below, this is still a one-layer model.

# In[4]:


# Build a simple Sequential model
model = tf.keras.Sequential([

    # Define the input shape
    tf.keras.Input(shape=(1,)),

    # Add a Dense layer
    tf.keras.layers.Dense(units=1)
    ])


# Now, you will compile the neural network. When you do so, you have to specify 2 functions: a [loss](https://keras.io/api/losses/) and an [optimizer](https://keras.io/api/optimizers/).
# 
# If you've seen lots of math for machine learning, here's where it's usually used. But in this case, it's nicely encapsulated in functions and classes for you. But what happens here?
# 
# You know that in the function declared at the start of this notebook, the relationship between the numbers is `y=2x-1`. When the computer is trying to 'learn' that, it makes a guess... maybe `y=10x+10`. The `loss` function measures the guessed answers against the known correct answers and measures how well or how badly it did.
# 
# It then uses the `optimizer` function to make another guess. Based on how the loss function went, it will try to minimize the loss. At that point maybe it will come up with something like `y=5x+5`, which, while still pretty bad, is closer to the correct result (i.e. the loss is lower).
# 
# It will repeat this for the number of _epochs_ which you will see shortly. But first, here's how you will tell it to use [mean squared error](https://keras.io/api/losses/regression_losses/#meansquarederror-function) for the loss and [stochastic gradient descent](https://keras.io/api/optimizers/sgd/) for the optimizer. You don't need to understand the math for these yet, but you can see that they work!
# 
# Over time, you will learn the different and appropriate loss and optimizer functions for different scenarios. 
# 

# In[5]:


# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')


# # Training the Neural Network
# 
# The process of training the neural network, where it 'learns' the relationship between the x's and y's is in the [`model.fit()`](https://keras.io/api/models/model_training_apis/#fit-method)  call. This is where it will go through the loop we spoke about above: making a guess, measuring how good or bad it is (aka the loss), using the optimizer to make another guess etc. It will do it for the number of `epochs` you specify. When you run this code, you'll see the loss on the right hand side.

# In[6]:


# Train the model
model.fit(xs, ys, epochs=500)


# Ok, now you have a model that has been trained to learn the relationship between `x` and `y`. You can use the [`model.predict()`](https://keras.io/api/models/model_training_apis/#predict-method) method to have it figure out the `y` for a previously unknown `x`. So, for example, if `x=10`, what do you think `y` will be? Take a guess before you run this code:

# In[7]:


# Make a prediction
print(f"model predicted: {model.predict(np.array([10.0]), verbose=0).item():.5f}")


# You might have thought `19`, right? But it ended up being a little under. Why do you think that is? 
# 
# Remember that neural networks deal with probabilities. So given the data that we fed the model with, it calculated that there is a very high probability that the relationship between `x` and `y` is `y=2x-1`, but with only 6 data points we can't know for sure. As a result, the result for 10 is very close to 19, but not necessarily 19.
# 
# As you work with neural networks, you'll see this pattern recurring. You will almost always deal with probabilities, not certainties, and will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.
# 

# In[ ]:




