import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn import svm

# making dataframe
df = pd.read_csv("input/twitter_sentiment_data.csv")
df.drop(columns=["tweetid"])

train, validate, test = np.split(df.sample(frac=1, random_state=0), [int(.6*len(df)), int(.8*len(df))])

if not os.path.isfile(f"model/config.json"):
    new_model = SentenceTransformer("all-MiniLM-L12-v2")
    new_model.save("model/")
transformer = SentenceTransformer.load("model/")

def preprocess_dataframe(dataframe):
    encoded = transformer.encode(dataframe["message"].values)
    dataframe["encoded"] = encoded.tolist()

preprocess_dataframe(train)
preprocess_dataframe(validate)

model = svm.SVC()
model.fit(train["encoded"].tolist(), train["sentiment"])
print(model.score(validate["encoded"].tolist(), validate["sentiment"]))

def remove_ats():
    return

def remove_rts():
    return


