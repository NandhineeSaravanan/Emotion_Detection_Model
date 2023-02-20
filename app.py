# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:10:04 2023

@author: sneka
"""

# import library
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression

app = FastAPI()
pickle_in = open("pipe_lr.pkl","rb")
pipe_lr=pickle.load(pickle_in)


#function to clean the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text


# Define the input data model
class InputData(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict-text")
async def predict(request: Request, input_data: InputData):
    # Get the text input from the input data
    cleaned_text = input_data.text
    
    # Make a prediction using the machine learning model
    prediction = pipe_lr.predict([cleaned_text])[0]
    
    # Return the predicted emotion as a JSON response
    return {'emotion': prediction}


    
    