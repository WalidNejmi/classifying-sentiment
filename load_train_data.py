import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np



if __name__ == '__main__':
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv')) #read the test data

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N,n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out the first five rows and last five rows
    tr_text_list = x_train_df['text'].values.tolist()
    rows = np.arange(0, 5)
    # for row_id in rows:
    #     text = tr_text_list[row_id]
    #     print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))

    # print("...")
    # rows = np.arange(N - 5, N)
    # for row_id in rows:
    #     text = tr_text_list[row_id]
    #     print("row %5d | y = %d | %s" % (row_id, y_train_df.values[row_id,0], text))


    # Make the vectorizer object that will grab the words from the x train data set
    vectorizer = CountVectorizer(lowercase=True, min_df=5, max_df=0.5, binary=False) #

    # Create the bag of words with the vectorizer for the x train and test dataset 
    x_train_bow = vectorizer.fit_transform(x_train_df['text'])
    x_test_bow = vectorizer.transform(x_test_df['text'])

    # Gets all the values of the is positive sentiment
    y_train_bow = y_train_df['is_positive_sentiment'].values

    # This function will make sure the proportion of different counts are the 
    # same across folds 
    cross_validation = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    # For the size
    # print(len(x_train_bow))

    param_grid = {
        'C': np.logspace(-4, 4, 25),  # More values for C
        'penalty': ['l2'],  # Different penalty types
    }

    grid_search = GridSearchCV(LogisticRegression(solver='lbfgs',random_state=12, max_iter=3000), param_grid, cv=cross_validation, scoring='roc_auc')
    grid_search.fit(x_train_bow, y_train_bow)
    
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    print(f"Best AUROC: {grid_search.best_score_}")

    # Train final model on full training data
    best_model.fit(x_train_bow, y_train_bow)

    # Predict on test set
    y_proba_test = best_model.predict_proba(x_test_bow)[:, 1]

    # Save predictions
    np.savetxt('yproba1_test.txt', y_proba_test)

    # # Get the keys (i.e., names of the metrics and parameters)
    # results = sorted(grid_search.cv_results_.keys())
    # print(results)  # This will print the list of keys (titles)

    # # Iterate through the keys and print their values
    # for key in results:
    #     print(f"Key: {key}, Values: {grid_search.cv_results_[key]}")

# Fined tuned for with english
# 0.8743472222222222

# Without english 
# 0.8801041666666667

    
    