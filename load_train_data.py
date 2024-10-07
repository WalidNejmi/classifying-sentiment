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
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

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
        # Preprocess text data
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_df=0.5, min_df=10)
    # print(vectorizer)
    X_train = vectorizer.fit_transform(x_train_df['text'])
    # print(X_train)
    X_test = vectorizer.transform(x_test_df['text'])
    print(X_train)
    y_train = y_train_df['is_positive_sentiment'].values

    # # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'C': np.logspace(-4, 4, 20),  # More values for C
        'penalty': ['l2', 'l1'],  # Different penalty types
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    print(f"Best AUROC: {grid_search.best_score_}")

    # Train final model on full training data
    best_model.fit(X_train, y_train)

    # Predict on test set
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    # Save predictions
    np.savetxt('yproba1_test.txt', y_proba_test)
