import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import re

# Define the path to the folder containing the files
neg_folder_path = 'C:/Users/ellio/OneDrive/Documents/Computer Science/NLP/neg'
pos_folder_path = 'C:/Users/ellio/OneDrive/Documents/Computer Science/NLP/pos'
# Initialize an empty array
neg_text_array = []
pos_text_array = []
negLabels = []
posLabels = []
# Use glob to find all text files in the folder
for file_path in glob.glob(os.path.join(neg_folder_path, '*.txt')):
    # Extract the rating from the filename using a regular expression
    # Assuming the filename format is something like 'movie_7.txt' where 7 is the rating
    match = re.search(r'_(\d)\.txt$', os.path.basename(file_path))
    if match:
        rating = int(match.group(1))
        negLabels.append(rating)
        
        # Open each file and read its content with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            neg_text_array.append(content)


for file_path in glob.glob(os.path.join(pos_folder_path, '*.txt')):
    # Open each file and read its content
    match = re.search(r'_(\d)\.txt$', os.path.basename(file_path))
    if match:
        rating = int(match.group(1))
        posLabels.append(rating)
        
        # Open each file and read its content with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            pos_text_array.append(content)

# Combine the two arrays into a single array
text_array = neg_text_array + pos_text_array

# Create an array of labels, where 0 represents negative sentiment and 1 represents positive sentiment
labels = negLabels + posLabels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_array, labels, test_size=0.2, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Print the sizes of the training, testing, and evaluation sets
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Evaluation set size:", len(X_eval))

print(X_train[0], y_train[0])
