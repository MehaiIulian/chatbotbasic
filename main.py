import json
import pickle
import random
from os.path import isfile
import nltk
import numpy
import tflearn
from nltk.stem import LancasterStemmer
import tensorflow as tf

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

# If input data (json file) has already been processed, open the file with the processed data
# If input data has not been processed before: create the necessary lists for the chatbot to train itself
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Loop through intents and extract the relevant data:
    # Turn each pattern into a list of words using nltk.word_tokenizer,
    # Then add each pattern into docs_x list and its associated tag into the docs_y list
    # Also, add all individual intents to the labels list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Get data ready to feed into our model with the help of stemming, and sort words and labels
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # Create "bag of words" to turn list of strings into numerical input for machine learning algorithm:
    # Create output lists which are the length of the amount of labels/tags we have in our dataset
    # Each position in the list will represent one distinct label/tag,
    # A 1 in any of those positions will show which label/tag is represented
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    # Convert training and output lists into numpy arrays
    training = numpy.array(training)
    output = numpy.array(output)

    # Save processed data, so that it does not have to be processed again later
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Setup of our model
tf.compat.v1.get_default_graph

# Define network, two hidden layers with eight neurons
# Input data with length of training data
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Define type of network (DNN) and take in designed network (from above)
model = tflearn.DNN(net)

# If model has already been fitted, load the model
# Otherwise: fit the model --> pass all of training data
if isfile("model.tflearn"):
    model.load("model.tflearn")

else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# Make predictions
# First step in classifying any sentences is to turn a sentence input of the user into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chatWithBot(inputText):
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        print("I didn't get that, try again")
        return "I didn't get that, try again"

    result = model.predict(numpyCurrentText[0:1])
    result_index = numpy.argmax(result)
    tag = labels[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        return random.choice(responses)

    else:
        print("I didn't get that, try again")
        return "I didn't get that, try again"


def chat():
    print("Start talking with the chatbot (try quit to stop)")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        print(chatWithBot(inp))

# chat()
