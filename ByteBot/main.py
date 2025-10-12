# Natural language processing and machine learning imports
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize stemmer for word normalization
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# Suppress TensorFlow warnings
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# Load intents data from JSON file
with open("intents.json") as file:
    data = json.load(file)

# Try to load preprocessed data from pickle file
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # If pickle file doesn't exist, preprocess the data
    words = []
    labels = []
    docs_x = []  # List of tokenized patterns
    docs_y = []  # List of corresponding tags

    # Extract words and labels from intents
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # Stem and normalize words, remove duplicates and punctuation
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    # Create empty output template
    out_empty = [0 for _ in range(len(labels))]

    # Create bag of words for each pattern
    for x, doc in enumerate(docs_x):
        bag = []

        # Stem the words in the current pattern
        wrds = [stemmer.stem(w) for w in doc]

        # Create bag of words: 1 if word exists, 0 otherwise
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Create output row with one-hot encoding for the label
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert to numpy arrays for training
    training = numpy.array(training)
    output = numpy.array(output)

    # Save preprocessed data to pickle file for faster loading next time
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Reset the default graph to avoid conflicts
tensorflow.compat.v1.reset_default_graph()

# Build neural network model
net = tflearn.input_data(shape=[None, len(training[0])])  # Input layer
net = tflearn.fully_connected(net, 8)  # First hidden layer with 8 neurons
net = tflearn.fully_connected(net, 8)  # Second hidden layer with 8 neurons
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # Output layer with softmax
net = tflearn.regression(net)  # Add regression layer

# Initialize the DNN model
model = tflearn.DNN(net)

# Try to load existing model, otherwise train a new one
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, multi_word):
    """
    Convert a sentence into a bag of words representation.

    Args:
        s: Input sentence string
        multi_word: List of all known words in vocabulary

    Returns:
        Numpy array representing bag of words (1 if word present, 0 otherwise)
    """
    bag = [0 for _ in range(len(multi_word))]

    # Tokenize and stem the input sentence
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # Mark which words from vocabulary are present in the sentence
    for se in s_words:
        for i, w in enumerate(multi_word):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    """
    Main chatbot function that handles user interaction.
    Continuously prompts for user input and generates responses.
    """
    print("Start talking with ByteBot! Type <quit> to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predict the intent of the user's input
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # Find the matching intent and get its responses
        responses = []
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                break

        # Print a random response from the matched intent
        if responses:
            print(random.choice(responses))
        else:
            print("I don't understand that.")

# Start the chatbot
chat()