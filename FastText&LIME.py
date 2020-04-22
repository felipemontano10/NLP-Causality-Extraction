

#Load the Packages

import pandas as pd 
import json
from pathlib import Path
import re
import random
import fasttext
import lime.lime_text
import numpy as np
import webbrowser
import io
import glob, os
import matplotlib.pyplot as plt

#Directory

directory = "/............./NLP/Project/Hypothesis classification"
os.chdir(directory)

#Open Data

data = pd.read_csv("training_set.csv")

#Prepare the DataSet to input in the FastText algorithm
reviews_data = Path("training_set.json")
fasttext_data = Path("fasttext_dataset.txt")
with open(reviews_data) as f:
    data=json.load(f)
output = fasttext_data.open("w") 
for line in data:
    hypothesis = line['label']
    sentence = line['sentence'].replace("\n", " ")

    fasttext_line = "__label__{} {}".format(hypothesis, sentence)

    output.write(fasttext_line + "\n")

#Data Exploratory Analysis
data_tok = []
size_sent = []
for row in data:
    data_tokenize = row["sentence"].split()
    data_tok.append(data_tokenize)
    size = len(data_tokenize)
    size_sent.append(size)
    
size_sent = pd.DataFrame(size_sent)     
size_sent = size_sent[size_sent[0]<60]
size_sent.plot.hist(grid=True, bins= 30, rwidth=0.9,
                   color='#607c8e', density = True, legend=None)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.text(50, 0.05, r'$\mu=18.5$', fontsize=12)
plt.text(50, 0.04, r'$\sigma=9.8$', fontsize=12)

size_sent.mean()
size_sent.std() 

#Split into Training and Test Set 

reviews_data = Path("training_set.json")
training_data = Path("fasttext_dataset_training.txt")
test_data = Path("fasttext_dataset_test.txt")

# What percent of data to save separately as test data
percent_test_data = 0.10

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

train_output=training_data.open("w")
test_output=test_data.open("w")
for line in data:
    hypothesis = line['label']
    sentence = line['sentence'].replace("\n", " ")
    sentence = strip_formatting(sentence)

    fasttext_line = "__label__{} {}".format(hypothesis, sentence)
    if random.random() <= percent_test_data:
        test_output.write(fasttext_line + "\n")
    else:
        train_output.write(fasttext_line + "\n")

#Define the model: fasttext
model = fasttext.train_supervised("fasttext_dataset_training.txt", wordNgrams = 1, lr = 0.3, dim = 120, loss="ns")

#Define the function that presents the results
#print(model.words)
print(model.labels)
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test('fasttext_dataset.txt'))

##LIME - This code was adapted from https://medium.com/@ageitgey/natural-language-processing-is-fun-part-3-explaining-model-predictions-486d8616813c

# This function regularizes a piece of text so it's in the same format
# that we used when training the FastText classifier.
def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

# LIME needs to be able to mimic how the classifier splits
# the string into words. So we'll provide a function that
# mimics how FastText works.
def tokenize_string(string):
    return string.split()

# Load our trained FastText classifier model (created in Part 2)
classifier = fasttext.train_supervised("fasttext_dataset_training.txt", wordNgrams = 1, lr = 0.3, dim = 120, loss="ns")

# Create a LimeTextExplainer. This object knows how to explain a text-based
# prediction by dropping words randomly.
explainer = lime.lime_text.LimeTextExplainer(
    # We need to tell LIME how to split the string into words. We can do this
    # by giving it a function to call to split a string up the same way FastText does it.
    split_expression=tokenize_string,
    # Our FastText classifer uses bigrams (two-word pairs) to classify text. Setting
    # bow=False tells LIME to not assume that our classifier is based on single words only.
    bow=False,
    # To make the output pretty, tell LIME what to call each possible prediction from our model.
    class_names=["No hypothesis", "Yes hypothesis"]
)

# LIME is designed to work with classifiers that generate predictions
# in the same format as Scikit-Learn. It expects every prediction to have
# a probability value for every possible label.
# The default FastText python wrapper generates predictions in a different
# format where it only returns the top N highest likelihood results. This
# code just calls the FastText predict function and then massages it into
# the format that LIME expects (so that LIME will work).
def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []
    # Ask FastText for the top 10 most likely labels for each piece of text.
    # This ensures we always get a probability score for every possible label in our model.
    labels, probabilities = classifier.predict(texts, 10)

    # For each prediction, sort the probabaility scores into the same order
    # (I.e. no_stars, 1_star, 2_star, etc). This is needed because FastText
    # returns predicitons sorted by most likely instead of in a fixed order.
    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs[order])

    return np.array(res)

## Review to explain
review_hypho = "environmental instability will be positively associated with strategic change."
review_non_hypho = "this difference remained significant in a logistic regression model with controls for firm size and industry"

# Pre-process the text of the review so it matches the training format
preprocessed_review = strip_formatting(review_random)

# Make a prediction and explain it!
exp = explainer.explain_instance(
    # The review to explain
    preprocessed_review,
    # The wrapper function that returns FastText predictions in scikit-learn format
    classifier_fn=lambda x: fasttext_prediction_in_sklearn_format(classifier, x),
    # How many labels to explain. We just want to explain the single most likely label.
    top_labels=1,
    # How many words in our sentence to include in the explanation. You can try different values.
    num_features=20,
)
# Save the explanation to an HTML file so it's easy to view.
# You can also get it to other formats: as_list(), as_map(), etc.
# See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation
output_filename = Path("/........../NLP/Project/Hypothesis classification").parent / "explanation_hypho_random.html"
exp.save_to_file(output_filename)

# Open the explanation html in our web browser.
webbrowser.open(output_filename.as_uri())





