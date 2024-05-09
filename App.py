import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

def preprocess_text_sklearn(text):

    # Define a regular expression pattern to match "RT" at the beginning of the string followed by any characters up to ":"
    pattern = r'^RT.*?:'

    # Use re.sub() to replace the matched pattern with an empty string
    text = re.sub(pattern, '', text)

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

# Get the same results each time
np.random.seed(0)

# Load the training data
print("\033[92m[+]Loading Training Data... \033[00m\n")

# Load data from CSV file
data = pd.read_csv("input/twitter_sentiment_data.csv")

# Define features
message_text = data["message"]
sentiment_value = (data["sentiment"]>0.5).astype(int)

# Break into training and test sets
message_train, message_test, y_train, y_test = train_test_split(message_text, sentiment_value, test_size=0.20, stratify=sentiment_value)

# Data preprocessing
preprocessed_tweets_sklearn = [preprocess_text_sklearn(message) for message in message_train]

# Tokenization
tokenized_message_data = [message.split() for message in message_text]

# Train Word2Vec embeddings
word2vec_model = Word2Vec(sentences=tokenized_message_data, vector_size=100, window=5, min_count=1, workers=4)

# Convert text data to sequences of word indices
max_sequence_length = max(len(tokens) for tokens in tokenized_message_data)
X_train_sequences = [[word2vec_model.wv.key_to_index[word] for word in tokens if word in word2vec_model.wv.key_to_index] for tokens in message_train]
X_test_sequences = [[word2vec_model.wv.key_to_index[word] for word in tokens if word in word2vec_model.wv.key_to_index] for tokens in message_test]

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Define CNN model
embedding_dim = 100
vocab_size = len(word2vec_model.wv)
num_filters = 64
kernel_size = 3

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluation
loss, accuracy = model.evaluate(X_test_padded, y_test)
print("Test Accurancy:", accuracy)

# Get vocabulary from training data
# vectorizer = CountVectorizer(stop_words='english')
# vectorizer.fit(preprocessed_tweets_sklearn)

# # Get word counts for training and test sets
# X_train = vectorizer.transform(preprocessed_tweets_sklearn)
# X_test = vectorizer.transform(message_test)

# Preview the dataset
# print("\033[92m[+]Data successfully loaded!\033[00m\n")
# print("\033[96mExample for Climate Change Being Man Made:\033[00m\n", message_train.iloc[27], "\n")
# print("\033[96mExample for Climate Change NOT Being Man Made:\033[00m\n", message_train.iloc[12], "\n")

# # Train a model and evaluate performance on test dataset
# classifier = LogisticRegression(max_iter=2000)
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print("\033[96mModel Performance Accuracy Against Test Data:\033[00m", score)

# # Function to classify any string
# def classify_string(string, investigate=False):
#     prediction = classifier.predict(vectorizer.transform([string]))[0]
#     if prediction == 0:
#         print("\033[92mClimate Change is Man Made: \033[00m", string)
#     else:
#         print("\033[91mClimate Chnage is NOT Man Made: \033[00m", string)

# test_tweet = input("\n\033[93mPlease enter a tweet: \033[00m")
# print("\n\033[96mModel's Decision: \033[00m")
# classify_string(test_tweet)

# coefficients = pd.DataFrame({"word": sorted(list(vectorizer.vocabulary_.keys())), "coeff": classifier.coef_[0]})
# print("\n\033[96mCoefficients: \033[00m")
# print(coefficients.sort_values(by=['coeff']).tail(10))

