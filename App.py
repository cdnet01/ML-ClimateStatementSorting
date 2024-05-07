import numpy as np
import torch
from torch.data import Field
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Get the same results each time
np.random.seed(0)

# Define pre-processing steps
TEXT = Field(tokenize='spacy', include_lengths=True, lower=True, stop_words='english')
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# Load the training data
print("\033[92m[+]Loading Training Data... \033[00m\n")

# Load data from CSV file
data = TabularDataset(
    path='input/twitter_sentiment_data.csv',
    format='csv',
    fields=[('message', TEXT),('sentiment', LABEL)]
)

# Define features
message_text = data["message"]
sentiment_value = (data["sentiment"]>0.5).astype(int)

# Split data into training and testing sets
train_data, test_data = data.split(split_ratio=0.8) # 80% training, 20% testing

# Build vocabulary
TEXT.build_vocab(train_data)

# Create iterators for batching and padding
BATCH_SIZE = 64 # hyperparameter that we can experiment with
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x : len(x.message),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Break into training and test sets
message_train, message_test, y_train, y_test = train_test_split(message_text, sentiment_value, test_size=0.30, stratify=sentiment_value)

# Get vocabulary from training data
vectorizer = CountVectorizer()
vectorizer.fit(message_train)

# Get word counts for training and test sets
X_train = vectorizer.transform(message_train)
X_test = vectorizer.transform(message_test)

# Preview the dataset
print("\033[92m[+]Data successfully loaded!\033[00m\n")
print("\033[96mExample for Climate Change Being Man Made:\033[00m\n", message_train.iloc[27], "\n")
print("\033[96mExample for Climate Change NOT Being Man Made:\033[00m\n", message_train.iloc[12], "\n")

# Train a model and evaluate performance on test dataset
classifier = LogisticRegression(max_iter=2000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("\033[96mModel Performance Accuracy Against Test Data:\033[00m", score)

# Function to classify any string
def classify_string(string, investigate=False):
    prediction = classifier.predict(vectorizer.transform([string]))[0]
    if prediction == 0:
        print("\033[92mClimate Change is Man Made: \033[00m", string)
    else:
        print("\033[91mClimate Chnage is NOT Man Made: \033[00m", string)

test_tweet = input("\n\033[93mPlease enter a tweet: \033[00m")
print("\n\033[96mModel's Decision: \033[00m")
classify_string(test_tweet)

coefficients = pd.DataFrame({"word": sorted(list(vectorizer.vocabulary_.keys())), "coeff": classifier.coef_[0]})
print("\n\033[96mCoefficients: \033[00m")
print(coefficients.sort_values(by=['coeff']).tail(10))

