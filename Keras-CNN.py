import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import SGD
from nltk.corpus import stopwords
import re
import ast

# Input Test Data
data = pd.read_csv("input/twitter_sentiment_data.csv")

# Remove tweets and labels where sentiment is equal to 2
filtered_message = [msg for msg, sent in zip(data["message"], data["sentiment"]) if sent != 2]
filtered_sentiment = [sent for sent in data["sentiment"] if sent != 2]

# Combine the filtered message and sentiment into a new dictionary
filtered_data = {"message": filtered_message, "sentiment": filtered_sentiment}

tweets = data["message"][:27733]
labels = data["sentiment"][:27733] # First 80% of input data

# Hyperparameters
max_words = 1000  # Maximum number of words in the vocabulary
maxlen = 100  # Maximum length of input sequences
embedding_dim = 100  # Dimension of word embeddings
filters = 100  # Number of filters in the convolutional layer
kernel_size = 5  # Size of convolutional kernels
pool_size = 4  # Size of max pooling window
hidden_dims = 64  # Number of neurons in the dense layer

# Tokenize the text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad sequences to ensure uniform length
X_train = pad_sequences(sequences, maxlen=maxlen)

# Create the CNN model
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    Conv1D(filters, kernel_size, activation='tanh'),
    MaxPooling1D(pool_size),
    Conv1D(filters, kernel_size, activation='tanh'),
    GlobalMaxPooling1D(),
    Dense(hidden_dims, activation='tanh'),
    Dense(1)  
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, labels, epochs=10, batch_size=32)

# Set the test data to be the last 20% of the input data
test_tweets = list(data["message"][-6933:])

# Tokenize and pad the test sequences
test_sequences = tokenizer.texts_to_sequences(test_tweets)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

# Make predictions on the test data
predictions = model.predict(X_test)

# Display the predictions
for i, tweet in enumerate(test_tweets):
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {predictions[i][0]}")

# Set test labels from the input data 
test_labels = list(data["sentiment"][-6933:])

# Calculate the accuracy
correct_predictions = 0
total_predictions = len(test_tweets)

for i, prediction in enumerate(predictions):
    if prediction < -0.33 and test_labels[i] < -0.33:
        correct_predictions += 1
    elif prediction >= -.33 and prediction <= 0.33 and test_labels[i] >= -0.33 and test_labels[i] <= 0.33:
        correct_predictions +=1
    elif prediction > 0.33 and test_labels[i] > 0.33:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2%}")
