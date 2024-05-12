import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense

# Example training data (tweet texts and labels)
tweets = [
    "Climate change is a serious threat to our planet.",
    "I don't believe in man-made climate change.",
    "The evidence for climate change is overwhelming.",
    "Climate change is a hoax created by scientists.",
    # Add more tweets...
]

data = pd.read_csv("input/twitter_sentiment_data.csv")

tweets = data["message"][:35154]

# tweets = data["message"][:10]

labels = np.array([2, -1, 2, -1])  # Example sentiment labels corresponding to tweets
labels = data["sentiment"][:35154]

# labels = data["sentiment"][:10]

# test_tweets = data["message"][-10:]
# test_labels = data["sentiment"][-10:]

# Hyperparameters
max_words = 1000  # Maximum number of words in the vocabulary
maxlen = 100  # Maximum length of input sequences
embedding_dim = 100  # Dimension of word embeddings
filters = 64  # Number of filters in the convolutional layer
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
    Conv1D(filters, kernel_size, activation='relu'),
    MaxPooling1D(pool_size),
    Conv1D(filters, kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(hidden_dims, activation='relu'),
    Dense(1)  # Output layer with one neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the validation data
# loss, accuracy = model.evaluate(test_tweets, test_labels)

# print("Validation Loss:", loss)
# print("Validation Accuracy:", accuracy)

# # Sample test inputs (tweet texts)
test_tweets = [
    "Climate change is a real problem that needs immediate action.",
    "I'm not convinced that humans are causing climate change.",
    "Global warming is just a natural cycle of the Earth.",
    # Add more test tweets...
]

test_tweets = list(data["message"][-8788:])

# Tokenize and pad the test sequences
test_sequences = tokenizer.texts_to_sequences(test_tweets)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

# Make predictions on the test data
predictions = model.predict(X_test)

# Display the predictions
for i, tweet in enumerate(test_tweets):
    print(f"Tweet: {tweet}")
    print(f"Predicted sentiment: {predictions[i][0]}")


# Sample test labels
test_labels = [2, 0, -1]  # Example test labels (-1, 0, and 1 indicate sentiments)

test_labels = list(data["sentiment"][-8788:])

# Calculate the accuracy
correct_predictions = 0
total_predictions = len(test_tweets)

for i, prediction in enumerate(predictions):
    predicted_sentiment = round(prediction[0])  # Round the prediction to the nearest integer
    if predicted_sentiment == test_labels[i]:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2%}")
