#!/usr/bin/env python
# coding: utf-8

# # Week 1: Explore the BBC News archive
# 
# Welcome! In this assignment you will be working with a variation of the [BBC News Classification Dataset](https://www.kaggle.com/c/learn-ai-bbc/overview), which contains 2225 examples of news articles with their respective categories.
# 
# 
# #### TIPS FOR SUCCESSFUL GRADING OF YOUR ASSIGNMENT:
# - All cells are frozen except for the ones where you need to submit your solutions or when explicitly mentioned you can interact with it.
# 
# - You can add new cells to experiment but these will be omitted by the grader, so don't rely on newly created cells to host your solution code, use the provided places for this.
# - You can add the comment # grade-up-to-here in any graded cell to signal the grader that it must only evaluate up to that point. This is helpful if you want to check if you are on the right track even if you are not done with the whole assignment. Be sure to remember to delete the comment afterwards!
# - Avoid using global variables unless you absolutely have to. The grader tests your code in an isolated environment without running all cells from the top. As a result, global variables may be unavailable when scoring your submission. Global variables that are meant to be used will be defined in UPPERCASE.
# - To submit your notebook, save it and then click on the blue submit button at the beginning of the page.
# 
# 
# Let's get started!

# In[1]:


import csv
import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:


import unittests


# Begin by looking at the structure of the csv that contains the data:

# In[3]:


with open("./data/bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"Each data point looks like this:\n\n{csvfile.readline()}")


# As you can see, each data point is composed of the category of the news article followed by a comma and then the actual text of the article.

# ## Exercise 1: parse_data_from_file
# 
# The csv is a very common format to store data and you will probably encounter it many times so it is good to be comfortable with it. Your first exercise will be to read the data from the raw csv file so you can analyze it and built models around it. To do so, complete the `parse_data_from_file` function below. 
# 
# Since this format is so common there are a lot of ways to deal with this files using python, both using the standard library or third party libraries such as [pandas](https://pandas.pydata.org/). Because of this the implementation details are entirely up to you, **the only requirement is that your function returns the `sentences` and `labels` as regular python lists**.
# 
# **Hints**:
# - Remember the file contains headers so take this into consideration.
# 
# 
# - If you are unfamiliar with  libraries such as pandas or numpy and you prefer to use python's standard library, take a look at [`csv.reader`](https://docs.python.org/3/library/csv.html#csv.reader), which lets you iterate over the lines of a csv file. 
# - You can use the [`read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) function from the pandas library.
# - You can use the [`loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) function from the numpy library.
# - If you use any of the two latter approaches remember you still need to convert the `sentences` and `labels` to regular python lists, so take a look at the docs to see how it can be done.
# 

# In[4]:


# GRADED FUNCTION: parse_data_from_file

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file
    
    Args:
        filename (str): path to the CSV file
    
    Returns:
        (list[str], list[str]): tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []

    ### START CODE HERE ###

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header if present

        for row in reader:
            label = row[0]
            sentence = row[1]
    
            labels.append(label)
            sentences.append(sentence)
	
    ### END CODE HERE ###

    return sentences, labels


# In[5]:


# Get sentences and labels as python lists
sentences, labels = parse_data_from_file("./data/bbc-text.csv")

print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words.\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}\n\n")


# ***Expected Output:***
# ```
# There are 2225 sentences in the dataset.
# 
# First sentence has 737 words.
# 
# There are 2225 labels in the dataset.
# 
# The first 5 labels are ['tech', 'business', 'sport', 'sport', 'entertainment']
# 
# ```

# In[6]:


# Test your code!
unittests.test_parse_data_from_file(parse_data_from_file)


# **An important note:**
# 
# At this point you typically would convert your data into a `tf.data.Dataset` (alternatively you could have used [tf.data.experimental.CsvDataset](https://www.tensorflow.org/api_docs/python/tf/data/experimental/CsvDataset) to do this directly but since this is an experimental feature it is better to avoid when possible) but for this assignment you will keep working with the data as regular python lists. 
# 
# The reason behind this is that by using a `tf.data.Dataset` some parts of this assignment will be much more difficult (in particular the next exercise), because when dealing with tensors you need to take additional considerations that you don't need to when dealing with lists and since this is the first assignment of the course, it is best to keep things simple. During next week's assignment you will get to see how this process looks like but for now carry on with the data in this format and worry not since TensorFlow is still compatible with these data formats!

# ## Exercise 2: standardize_func
# 
# One important step when working with text data is to standardize it so it is easier to extract information out of it. For instance, you probably want to convert it all to lower-case (so the same word doesn't have different representations such as "hello" and "Hello") and to remove the [stopwords](https://en.wikipedia.org/wiki/Stop_word) from it. These are the most common words in the language and they rarely provide useful information for the classification process. The next cell provides a list of common stopwords which you can use in the exercise:
# 
# 
# 

# In[7]:


# List of stopwords
STOPWORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# To achieve this, complete the `standardize_func` below. This function should receive a string and return another string that excludes all of the stopwords provided from it, as well as converting it to lower-case. 
# 
# **Hints:**
# - You only need to account for whitespace as the separation mechanism between words in the sentence.
# 
# - The list of stopwords is already provided for you as a global variable you can safely use.
# 
# - Check out the [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) method for python strings.
# 
# - The returned sentence should not include extra whitespace so the string "hello&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;again&nbsp;&nbsp;&nbsp;FRIENDS" should be standardized to "hello friends".

# In[14]:


# GRADED FUNCTION: standardize_func

def standardize_func(sentence):
    """Standardizes sentences by converting to lower-case and removing stopwords.

    Args:
        sentence (str): Original sentence.

    Returns:
        str: Standardized sentence in lower-case and without stopwords.
    """
    
    ### START CODE HERE ###
    sentence = sentence.lower()
    words = sentence.split()
    filteredData = [w for w in words if w not in STOPWORDS]
    sentence = " ".join(filteredData)
	
    ### END CODE HERE ###

    return sentence


# In[15]:


test_sentence = "Hello! We're just about to see this function in action =)"
standardized_sentence = standardize_func(test_sentence)
print(f"Original sentence is:\n{test_sentence}\n\nAfter standardizing:\n{standardized_sentence}")

standard_sentences = [standardize_func(sentence) for sentence in sentences]

print("\n\n--- Apply the standardization to the dataset ---\n")
print(f"There are {len(standard_sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words originally.\n")
print(f"First sentence has {len(standard_sentences[0].split())} words (after removing stopwords).\n")


# ***Expected Output:***
# ```
# Original sentence is:
# Hello! We're just about to see this function in action =)
# 
# After standardizing:
# hello! just see function action =)
# 
# 
# --- Apply the standardization to the dataset ---
# 
# There are 2225 sentences in the dataset.
# 
# First sentence has 737 words originally.
# 
# First sentence has 436 words (after removing stopwords).
# 
# ```

# In[16]:


# Test your code!
unittests.test_standardize_func(standardize_func)


# With the dataset standardized you could go ahead and convert it to a `tf.data.Dataset`, which you will NOT be doing for this assignment. However if you are curious, this can be achieved like this:
# 
# ```python
# dataset = tf.data.Dataset.from_tensor_slices((standard_sentences, labels))
# ```

# ## Exercise 3: fit_vectorizer
# 
# Now that your data is standardized, it is time to vectorize the sentences of the dataset. For this complete the `fit_vectorizer` below. 
# 
# This function should receive the list of sentences as input and return a [`tf.keras.layers.TextVectorization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) that has been adapted to those sentences.

# In[17]:


# GRADED FUNCTION: fit_vectorizer

def fit_vectorizer(sentences):
    """
    Instantiates the TextVectorization layer and adapts it to the sentences.
    
    Args:
        sentences (list[str]): lower-cased sentences without stopwords
    
    Returns:
        tf.keras.layers.TextVectorization: an instance of the TextVectorization layer adapted to the texts.
    """

    ### START CODE HERE ###

    # Instantiate the TextVectorization class
    vectorizer = tf.keras.layers.TextVectorization()

    # Adapt to the sentences
    vectorizer.adapt(sentences)

    ### END CODE HERE ###

    return vectorizer


# In[18]:


# Create the vectorizer adapted to the standardized sentences
vectorizer = fit_vectorizer(standard_sentences)

# Get the vocabulary
vocabulary = vectorizer.get_vocabulary()

print(f"Vocabulary contains {len(vocabulary)} words\n")
print("[UNK] token included in vocabulary" if "[UNK]" in vocabulary else "[UNK] token NOT included in vocabulary")


# ***Expected Output:***
# ```
# Vocabulary contains 33088 words
# 
# [UNK] token included in vocabulary
# 
# ```

# In[19]:


# Test your code!
unittests.test_fit_vectorizer(fit_vectorizer)


# Next, you can use the adapted vectorizer to vectorize the sentences in your dataset. Notice that by default `tf.keras.layers.TextVectorization` pads the sequences so all of them have the same length (typically the length of the longest sentence will be used if no truncation is defined), this is important because neural networks expect the inputs to have the same size.

# In[20]:


# Vectorize and pad sentences
padded_sequences = vectorizer(standard_sentences)

# Show the output
print(f"First padded sequence looks like this: \n\n{padded_sequences[0]}\n")
print(f"Tensor of all sequences has shape: {padded_sequences.shape}\n")
print(f"This means there are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}")


# Notice that now the variable refers to `sequences` rather than `sentences`. This is because all your text data is now encoded as a sequence of integers.

# ## Exercise 4: fit_label_encoder
# 
# 
# With the sentences already vectorized it is time to encode the labels so they can also be fed into a neural network. For this complete the `fit_label_encoder` below. 
# 
# This function should receive the list of labels as input and return a [`tf.keras.layers.StringLookup`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup) that has been adapted to those sentences. In theory you could also use `tf.keras.layers.TextVectorization` layer here but it provides a lot of extra functionality that is not required so it ends up being overkill. `tf.keras.layers.StringLookup` is able to perform the job just fine and it is much simpler.
# 
# **Hints:**
# - Since all of the texts have their corresponding labels you need to ensure that the vocabulary does not include the out-of-vocabulary (OOV) token since that is not a valid label.

# In[30]:


# GRADED FUNCTION: fit_label_encoder

def fit_label_encoder(labels):
    """
    Tokenizes the labels
    
    Args:
        labels (list[str]): labels to tokenize
    
    Returns:
        tf.keras.layers.StringLookup: adapted encoder for labels
    """
    ### START CODE HERE ###
    
    # Instantiate the StringLookup layer. Remember that you don't want any OOV tokens!
    label_encoder = tf.keras.layers.StringLookup(num_oov_indices=0)
    
    # Adapt the StringLookup layer to the labels
    label_encoder.adapt(labels)
	
    ### END CODE HERE ###
    
    return label_encoder


# In[31]:


# Create the encoder adapted to the labels
label_encoder = fit_label_encoder(labels)

# Get the vocabulary
vocabulary = label_encoder.get_vocabulary()

# Encode labels
label_sequences = label_encoder(labels)

print(f"Vocabulary of labels looks like this: {vocabulary}\n")
print(f"First ten labels: {labels[:10]}\n")
print(f"First ten label sequences: {label_sequences[:10]}\n")


# ***Expected Output:***
# ```
# Vocabulary of labels looks like this: ['sport', 'business', 'politics', 'tech', 'entertainment']
# 
# First ten labels: ['tech', 'business', 'sport', 'sport', 'entertainment', 'politics', 'politics', 'sport', 'sport', 'entertainment']
# 
# First ten label sequences: [3 1 0 0 4 2 2 0 0 4]
# 
# ```

# You should see that each encoded label corresponds to the index of its corresponding label in the vocabulary!

# In[32]:


# Test your code!
unittests.test_fit_label_encoder(fit_label_encoder)


# Great job! Now you have successfully performed all the necessary steps to train a neural network capable of processing text. This is all for now but in next week's assignment you will train a model capable of classifying the texts in this same dataset!
# 
# 
# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented functions to process various text data processing ranging from pre-processing, reading from raw files and tokenizing text.
# 
# **Keep it up!**
