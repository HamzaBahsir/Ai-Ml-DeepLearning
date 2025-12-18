#!/usr/bin/env python
# coding: utf-8

# # Week 4: Multi-class Classification
# 
# Welcome to the last assignment! In this notebook, you will get a chance to work on a multi-class classification problem. You will be using the [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset, which contains 28x28 images of hands depicting the letters of the english alphabet. 
# 
# #### TIPS FOR SUCCESSFUL GRADING OF YOUR ASSIGNMENT:
# 
# - All cells are frozen except for the ones where you need to submit your solutions or when explicitly mentioned you can interact with it.
# 
# 
# - You can add new cells to experiment but these will be omitted by the grader, so don't rely on newly created cells to host your solution code, use the provided places for this.
# 
# - You can add the comment # grade-up-to-here in any graded cell to signal the grader that it must only evaluate up to that point. This is helpful if you want to check if you are on the right track even if you are not done with the whole assignment. Be sure to remember to delete the comment afterwards!
# 
# - Avoid using global variables unless you absolutely have to. The grader tests your code in an isolated environment without running all cells from the top. As a result, global variables may be unavailable when scoring your submission. Global variables that are meant to be used will be defined in UPPERCASE.
# 
# - To submit your notebook, save it and then click on the blue submit button at the beginning of the page.
# 
# Let's get started!

# In[1]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


import unittests


# In this assignment you will actually be working with a modified version of the original [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)  dataset. The original dataset is presented as a csv file, however this makes the pre processing of the data very different from what you have been doing so far. To make loading the images and creating the datasetss more aligned with what you have learned so far, we have already downloaded each image as a .png file. You can find them in the `data/train` and `data/validation` folders. As the names suggest, the images in the first folder will be used for training, and the ones in the latter will be used for validation. 
# 
# Begin by defining some globals with the paths to the training and test folders.

# In[3]:


TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'


# Let's explore the `./data` folder containing the images. There is a subdirectory for each class. In this case there will be 24 folders one for each letter in the alphabet, except for letters J and Z. Because of gesture motions these two letters can't be represented by an image, and are thus not included on the dataset.
# 
# The complete tree looks like this:
# 
# ```
# .└── data/
#     ├── train/
#     |    ├── A/
#     |    │   ├── a1.jpg
#     |    │   ├── a2.jpg
#     |    │   └── ...
#     |    ├── B/
#     |        ├── b1.jpg
#     |        ├── b2.jpg
#     |        └── ...
#     |    ├── ...
#     |    ├── I/
#     |    |    ├── i1.jpg
#     |    |    ├── i2.jpg
#     |    |    └── ...
#     |    ├── K/
#     |    |    ├── k1.jpg
#     |    |    ├── k2.jpg
#     |    |    └── ...
#     |    ├── ...
#     |    └── Y/
#     |        ├── y1.jpg
#     |        ├── y2.jpg
#     |        └── ...
#     └── validation/
#          ├── A/
#          │   ├── a1.jpg
#          │   ├── a2.jpg
#          │   └── ...
#          ├── B/
#          |   ├── b1.jpg
#          |   ├── b2.jpg
#          |   └── ...
#          ├── ...
#          ├── I/
#          |    ├── i1.jpg
#          |    ├── i2.jpg
#          |    └── ...
#          ├── K/
#          |    ├── k1.jpg
#          |    ├── k2.jpg
#          |    └── ...
#          ├── ...
#          └── Y/
#              ├── y1.jpg
#              ├── y2.jpg
#              └── ...
# ```

# Let's take a look at what the images look like.

# In[4]:


fig, axes = plt.subplots(1, 6, figsize=(14, 3))
fig.suptitle('Sign Language MNIST Images', fontsize=16)

# Plot one image from the first 4 letters
for ii, letter in enumerate(['A' , 'B', 'C', 'D', 'E', 'F']):
    dir = f'./data/train/{letter}'
    img = tf.keras.preprocessing.image.load_img(dir+'/'+os.listdir(dir)[0])
    axes[ii].imshow(img)
    axes[ii].set_title(f'Example of letter {letter}')


# By plotting the images with `matplotlib` you can readily see images have a resolution of 28x28 (look at the image axes) and are in greyscale, but you can double check this by using the code below:

# In[5]:


# Convert the image into its numpy array representation
sample_array = tf.keras.preprocessing.image.img_to_array(img)

print(f"Each image has shape: {sample_array.shape}")


# Don't worry about the last dimension. That is because the [`img_to_array`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array) function returns a 3D array. You can easily check that actually it has repeated the same values in each dimension, for example, take a look at the first 5 columns of the image. All you really care about is that your image is 28x28 pixels. 

# In[6]:


sample_array[0,:5]


# ## Creating the datasets for the CNN
# 
# ### Exercise 1: train_val_datasets
# 
# Your first task is to code the function that will create the datasets that will yield batches of images, both for training and validation. For this complete the `train_val_datasets` function below.
# 
# For grading purposes, make sure to use a **batch size of 32**. 

# In[7]:


# GRADED FUNCTION: train_val_datasets
def train_val_datasets():
    """Create train and validation datasets

    Returns:
        (tf.data.Dataset, tf.data.Dataset): train and validation datasets
    """
    ### START CODE HERE ###
    train_dataset = tf.keras.utils.image_dataset_from_directory( 
        directory=TRAIN_DIR,
        batch_size=32,
        image_size=(28,28),
		label_mode='categorical',
        color_mode = "grayscale", # Use this argument to get just one color dimension, because it is greyscale 
    ) 
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory( 
        directory=VALIDATION_DIR,
        batch_size=32,
        image_size=(28,28),
		label_mode='categorical',
        color_mode = "grayscale", # Use this argument to get just one color dimension, because it imgs are greyscale 
    ) 
    ### END CODE HERE ###
    
    return train_dataset, validation_dataset


# In[8]:


# Create train and validation datasets
train_dataset, validation_dataset = train_val_datasets()
print(f"Images of train dataset have shape: {train_dataset.element_spec[0].shape}")
print(f"Labels of train dataset have shape: {train_dataset.element_spec[1].shape}")
print(f"Images of validation dataset have shape: {validation_dataset.element_spec[0].shape}")
print(f"Labels of validation dataset have shape: {validation_dataset.element_spec[1].shape}")   


# **Expected Output:**
# ```
# Found 27455 files belonging to 24 classes.
# Found 7173 files belonging to 24 classes.
# Images of train dataset have shape: (None, 28, 28, 1)
# Labels of train dataset have shape: (None, 24)
# Images of validation dataset have shape: (None, 28, 28, 1)
# Labels of validation dataset have shape: (None, 24)
# ```

# In[9]:


# Test your function
unittests.test_train_val_datasets(train_val_datasets)


# ## Coding the CNN
# 
# ### Exercise : create_model
# 
# One last step before training is to define the architecture of the model that will be trained.
# 
# Complete the `create_model` function below. This function should return a Keras' model that uses the [`Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) API.
# 
# A couple of things to keep in mind when defining the architecture:
# 
# - Start the model with an [`Input`](https://www.tensorflow.org/api_docs/python/tf/keras/Input) followed by a layer that rescales your images so that each pixel has values between 0 and 1 
#   
# - There different ways to implement the output layer, however, we expect the last layer of your model to have a number of units that corresponds to the number of possible categories, as well as the correct activation function.
# 
# - Aside from defining the architecture of the model, you should also compile it so make sure to use a `loss` function that is suitable for multi-class classification. Remember to also define suitable `metric` to monitor.
# 
# **Note that you should use no more than 2 Conv2D and 2 MaxPooling2D layers to achieve the desired performance. You can also add dropout layers to improve training**

# In[43]:


# GRADED FUNCTION: create_model
def create_model():
    """Create the classifier model

    Returns:
        tf.keras.model.Sequential: CNN for multi-class classification
    """
    ### START CODE HERE ###      
    
    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = tf.keras.models.Sequential([ 
        # Define an input layer
        tf.keras.Input(shape=(28,28,1)), # Set correct input size
        # Rescale images
        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(24,activation='softmax'),
        ]) 

    model.compile(optimizer = tf.optimizers.Adam(),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    ### END CODE HERE ### 
    return model


# The next cell allows you to check the number of total and trainable parameters of your model and prompts a warning in case these exceeds those of a reference solution, this serves the following 3 purposes listed in order of priority:
# 
# - Helps you prevent crashing the kernel during training.
# 
# - Helps you avoid longer-than-necessary training times.
# - Provides a reasonable estimate of the size of your model. In general you will usually prefer smaller models given that they accomplish their goal successfully.
# 
# 
# **Notice that this is just informative** and may be very well below the actual limit for size of the model necessary to crash the kernel. So even if you exceed this reference you are probably fine. However, **if the kernel crashes during training or it is taking a very long time and your model is larger than the reference, come back here and try to get the number of parameters closer to the reference.**

# In[44]:


# Create your model
model = create_model()

# Check parameter count against a reference solution
unittests.parameter_count(model)


# Check that the input and output shape of your model are correct

# In[45]:


print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')


# Expected output:
# 
# ```
# Input shape: (None, 28, 28, 1)
# Output shape: (None, 24)
# ```

# Using the `summary` method you can visulize the model you just defined.

# In[46]:


model.summary()


# 
# Check that the architecture you used is compatible with the dataset (you can ignore the warnings prompted by using the GPU):

# In[47]:


for images, labels in train_dataset.take(1):
	example_batch_images = images
	example_batch_labels = labels
	
try:
	model.evaluate(example_batch_images, example_batch_labels, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function, last layer and label_mode are compatible with one another.")
else:
	predictions = model.predict(example_batch_images, verbose=False)
	print(f"predictions have shape: {predictions.shape}")


# **Expected output:**
# 
# ```
# predictions have shape: (32, 24)
# ```

# In[48]:


# Test your function
unittests.test_create_model(create_model)


# Finally, you can go ahead and train your model

# In[49]:


# Train your model
history = model.fit(train_dataset,
                    epochs=15,
                    validation_data=validation_dataset)


# Now take a look at your training history:

# In[51]:


# Get training and validation accuracies
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Training and validation accuracy')

for i, (data, label) in enumerate(zip([(acc, val_acc), (loss, val_loss)], ["Accuracy", "Loss"])):
    ax[i].plot(epochs, data[0], 'r', label="Training " + label)
    ax[i].plot(epochs, data[1], 'b', label="Validation " + label)
    ax[i].legend()
    ax[i].set_xlabel('epochs')

plt.show()


# You will not be graded based on the accuracy of your model but try making it as high as possible for both training and validation, as an optional exercise, **after submitting your notebook for grading**.
# 
# A reasonable benchmark is to achieve over 99% accuracy for training and over 95% accuracy for validation within 15 epochs. Try tweaking your model's architecture or the augmentation techniques to see if you can achieve these levels of accuracy.

# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented a convolutional neural network that is able to perform multi-class classification tasks! Nice job!
# 
# **Keep it up!**
