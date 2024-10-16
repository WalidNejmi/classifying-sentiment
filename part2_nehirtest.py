from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from load_train_data import load_data
import numpy as np


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# def main():
#     x_train_df, y_train_df, x_test_df = load_data()
    

#     # Run for Roberta Model
#     # encoded_x_train = tokenizer(x_train_df, return_tensors="pt")
#     # encoded_y_train = tokenizer(y_train_df, return_tensors="pt")
#     encoded_x_train = tokenizer(x_train_df['text'].tolist(), return_tensors="pt", padding=True, truncation=True)
#     encoded_y_train = tokenizer(y_train_df['text'].tolist(), return_tensors="pt", padding=True, truncation=True)

#     output = model(**encoded_x_train)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     print(scores)

# if __name__ == '__main__':
#     main()

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    # Check the structure of the DataFrames
    print("x_train_df columns:", x_train_df.columns)
    print("y_train_df columns:", y_train_df.columns)

    # Ensure x_train_df and y_train_df are in the correct format
    max_length = 128  # Set a max_length for truncation

    # It shouldn't be the train data, it should be the test data!!
    encoded_x_train = tokenizer(x_train_df['text'].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    # encoded_y_train = tokenizer(y_train_df['text'].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    print("here")
    output = model(**encoded_x_train)
    logits = output.logits  # Access the logits directly
    scores = softmax(logits.detach().numpy(), axis=1)  # Ensure to use axis=1 for 2D arrays
    print(scores)

if __name__ == "__main__":
    main()

















