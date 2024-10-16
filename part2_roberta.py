from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
from scipy.special import softmax
from load_train_data import load_data
import numpy as np

# This gets the encoded text using the tokenizer, and then ran the model on
# the text
def polarity_scores_roberta(text):  
    encoded_text = tokenizer(text, return_tensors='pt')
    outputs = model(**encoded_text)
    scores = outputs[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'neg': scores[0], 'neu': scores[1], 'pos': scores[2]}
    return scores_dict

# This just grabs the positive score
def get_positive_score(text):
        scores = polarity_scores_roberta(text)
        return scores['pos']

if __name__ == "__main__":
    x_train_df, y_train_df, x_test_df = load_data()

    X_train = x_train_df['text'] 
    y_train = y_train_df['is_positive_sentiment']
    X_test = x_test_df['text'] 
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Function to get positive sentiment score
    # Apply get positive score function to each sentence in X_test and get 
    # probabilities
    y_proba_test = X_test.apply(lambda text: get_positive_score(text))

    # Save the probabilities to a file
    np.savetxt('yprobaba_roberta.txt', y_proba_test)

    # Load the dataset
    # data = pd.read_csv('data.csv')



