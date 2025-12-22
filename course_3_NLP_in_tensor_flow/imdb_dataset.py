#!/usr/bin/env python
# coding: utf-8

# # Ungraded Lab: Training a binary classifier with the IMDB Reviews Dataset
# 
# In this lab, you will be building a sentiment classification model to distinguish between positive and negative movie reviews. You will train it on the [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) dataset and visualize the word embeddings generated after training.
# 

# ## Imports
# 
# As usual, you will start by importing the necessary packages.

# In[1]:


import tensorflow_datasets as tfds
import tensorflow as tf
import io


# ## Download the Dataset
# 
# You will load the dataset via [Tensorflow Datasets](https://www.tensorflow.org/datasets), a collection of prepared datasets for machine learning. If you're running this notebook on your local machine, make sure to have the [`tensorflow-datasets`](https://pypi.org/project/tensorflow-datasets/) package installed before importing it. You can install it via `pip` as shown in the commented cell below.

# In[2]:


# Install this package if running on your local machine
# !pip install -q tensorflow-datasets


# The [`tfds.load()`](https://www.tensorflow.org/datasets/api_docs/python/tfds/load) method downloads the dataset into your working directory. You can set the `with_info` parameter to `True` if you want to see the description of the dataset. The `as_supervised` parameter, on the other hand, is set to load the data as `(input, label)` pairs.
# 
# To ensure smooth operation, the data was pre-downloaded and saved in the data folder. When you have the data already downloaded, you can read it by passing two additional arguments. With ` data_dir="./data/"`, you specify the folder where the data is located (if different than default) and by setting `download=False` you explicitly tell the method to read the data from the folder, rather than downloading it.

# In[3]:


# Load the IMDB Reviews dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True, data_dir="./data/", download=False)


# In[4]:


# Print information about the dataset
print(info)


# As you can see in the output above, there is a total of 100,000 examples in the dataset and it is split into `train`, `test` and `unsupervised` sets. For this lab, you will only use `train` and `test` sets because you will need labeled examples to train your model.

# ## Split the dataset
# 
# The `imdb` dataset that you downloaded earlier contains a dictionary pointing to [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects.

# In[5]:


# Print the contents of the dataset
print(imdb)


# You can preview the raw format of a few examples by using the [`take()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take) method and iterating over it as shown below:

# In[6]:


# View 4 training examples
for example in imdb['train'].take(4):
  print(example)


# You can see that each example is a 2-element tuple of tensors containing the text first, then the label (shown in the `numpy()` property). The next cell below will take all the `train` and `test` sentences and labels into separate lists so you can preprocess the text and feed it to the model later.

# In[7]:


# Get the train and test sets
train_dataset, test_dataset = imdb['train'], imdb['test']


# ## Generate Padded Sequences
# 
# Now you can do the text preprocessing steps you've learned last week. You will convert the strings to integer sequences, then pad them to a uniform length. The parameters are separated into its own code cell below so it will be easy for you to tweak it later if you want.

# In[8]:


# Parameters

VOCAB_SIZE = 10000
MAX_LENGTH = 120
EMBEDDING_DIM = 16
PADDING_TYPE = 'pre'
TRUNC_TYPE = 'post'


# An important thing to note here is you should generate the vocabulary based only on the training set. You should not include the test set because that is meant to represent data that the model hasn't seen before. With that, you can expect more unknown tokens (i.e. the value `1`) in the integer sequences of the test data. Also for clarity in demonstrating the transformations, you will first separate the reviews and labels. You will see other ways to implement the data pipeline in the next labs.

# In[9]:


# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
)

# Get the string inputs and integer outputs of the training set
train_reviews = train_dataset.map(lambda review, label: review)
train_labels = train_dataset.map(lambda review, label: label)

# Get the string inputs and integer outputs of the test set
test_reviews = test_dataset.map(lambda review, label: review)
test_labels = test_dataset.map(lambda review, label: label)

# Generate the vocabulary based only on the training set
vectorize_layer.adapt(train_reviews)


# You will define a padding function to generate the padded sequences. Note that the `pad_sequences()` function expects an iterable (e.g. list) while the input to this function is a `tf.data.Dataset`. Here's one way to do the conversion:
# * Put all the elements in a single [ragged batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#ragged_batch) (i.e. batch with elements that have different lengths).
#     * You will need to specify the batch size and it has to match the number of all elements in the dataset. From the output of the dataset info earlier, you know that this should be 25000.
#     * Instead of specifying a specific number, you can also use the [cardinality()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality) method. This computes the number of elements in a `tf.data.Dataset`.
# * Use the [get_single_element()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#get_single_element) method on the single batch to output a Tensor.
# * Convert back to a `tf.data.Dataset`. You'll see why this is needed in the next cell.

# In[10]:


def padding_func(sequences):
  '''Generates padded sequences from a tf.data.Dataset'''

  # Put all elements in a single ragged batch
  sequences = sequences.ragged_batch(batch_size=sequences.cardinality())

  # Output a tensor from the single batch
  sequences = sequences.get_single_element()

  # Pad the sequences
  padded_sequences = tf.keras.utils.pad_sequences(sequences.numpy(), 
                                                  maxlen=MAX_LENGTH, 
                                                  truncating=TRUNC_TYPE, 
                                                  padding=PADDING_TYPE
                                                 )

  # Convert back to a tf.data.Dataset
  padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

  return padded_sequences


# This is the pipeline to convert the raw string inputs to padded integer sequences:
# * Use the [map()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) method to pass each string to the `TextVectorization` layer defined earlier.
# * Use the [apply()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#apply) method to use the padding function on the entire dataset.
# 
# The difference between `map()` and `apply()` is the mapping function in `map()` expects its input to be single elements (i.e. element-wise transformations), while the transformation function in `apply()` expects its input to be the entire dataset in the pipeline.

# In[11]:


# Apply the layer to the train and test data
train_sequences = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
test_sequences = test_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)


# You can take a few examples from the results and observe that the raw strings are now converted to padded integer sequences.

# In[12]:


# View 2 training sequences
for example in train_sequences.take(2):
  print(example)
  print()


# You will now re-combine the sequences with the labels to prepare it for training.

# In[13]:


train_dataset_vectorized = tf.data.Dataset.zip(train_sequences,train_labels)
test_dataset_vectorized = tf.data.Dataset.zip(test_sequences,test_labels)


# In[14]:


# View 2 training sequences and its labels
for example in train_dataset_vectorized.take(2):
  print(example)
  print()


# Lastly, you will optimize and batch the datasets.

# In[15]:


SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Optimize the datasets for training
train_dataset_final = (train_dataset_vectorized
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       )

test_dataset_final = (test_dataset_vectorized
                      .cache()
                      .prefetch(PREFETCH_BUFFER_SIZE)
                      .batch(BATCH_SIZE)
                      )


# ## Build and Compile the Model
# 
# With the data already preprocessed, you can proceed to building your sentiment classification model. The input will be an [`Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) layer. The main idea here is to represent each word in your vocabulary with vectors. These vectors have trainable weights so as your neural network learns, words that are most likely to appear in a positive tweet will converge towards similar weights. Similarly, words in negative tweets will be clustered more closely together. You can read more about word embeddings [here](https://www.tensorflow.org/text/guide/word_embeddings).
# 
# After the `Embedding` layer, you will flatten its output and feed it into a `Dense` layer. You will explore other architectures for these hidden layers in the next labs.
# 
# The output layer would be a single neuron with a sigmoid activation to distinguish between the 2 classes. As is typical with binary classifiers, you will use the `binary_crossentropy` as your loss function while training.

# In[16]:


# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Setup the training parameters
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Print the model summary
model.summary()


# ## Train the Model
# 
# Next, of course, is to train your model. With the current settings, you will get near perfect training accuracy after just 5 epochs but the validation accuracy will only be at around 80%. See if you can still improve this by adjusting some of the parameters earlier (e.g. the `VOCAB_SIZE`, number of `Dense` neurons, number of epochs, etc.).

# In[17]:


NUM_EPOCHS = 5

# Train the model
model.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)


# ## Visualize Word Embeddings
# 
# After training, you can visualize the trained weights in the `Embedding` layer to see words that are clustered together. The [Tensorflow Embedding Projector](https://projector.tensorflow.org/) is able to reduce the 16-dimension vectors you defined earlier into fewer components so it can be plotted in the projector. First, you will need to get these weights and you can do that with the cell below:

# In[18]:


# Get the embedding layer from the model (i.e. first layer)
embedding_layer = model.layers[0]

# Get the weights of the embedding layer
embedding_weights = embedding_layer.get_weights()[0]

# Print the shape. Expected is (vocab_size, embedding_dim)
print(embedding_weights.shape)


# You will need to generate two files:
# 
# * `vecs.tsv` - contains the vector weights of each word in the vocabulary
# * `meta.tsv` - contains the words in the vocabulary

# You will get the word list from the `TextVectorization` layer you adapted earler, then start the loop to generate the files. You will loop `vocab_size-1` times, skipping the `0` key because it is just for the padding.

# In[19]:


# Open writeable files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

# Get the word list
vocabulary = vectorize_layer.get_vocabulary()

# Initialize the loop. Start counting at `1` because `0` is just for the padding
for word_num in range(1, len(vocabulary)):

  # Get the word associated withAttributeError the current index
  word_name = vocabulary[word_num]

  # Get the embedding weights associated with the current index
  word_embedding = embedding_weights[word_num]

  # Write the word name
  out_m.write(word_name + "\n")

  # Write the word embedding
  out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

# Close the files
out_v.close()
out_m.close()


# You can find the files in your current working directory and download them. Now you can go to the [Tensorflow Embedding Projector](https://projector.tensorflow.org/) and load the two files you downloaded to see the visualization. You can search for words like `worst` and `fantastic` and see the other words closely located to these.

# ## Wrap Up
# 
# In this lab, you were able build a simple sentiment classification model and train it on preprocessed text data. In the next lessons, you will revisit the Sarcasm Dataset you used in Week 1 and build a model to train on it.

# In[ ]:




