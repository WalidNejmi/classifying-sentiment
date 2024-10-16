import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from load_train_data import load_data

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    # Create a pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=True, binary=False)),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=5000, random_state=12, penalty='l2'))
    ])

    y_train_bow = y_train_df['is_positive_sentiment'].values

    cross_validation = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)

    param_grid = {
        'vectorizer__max_df': np.linspace(0.3, 0.99, 7),
        'vectorizer__min_df': range(1, 6),
        'classifier__C': np.logspace(-4, 4, 25),
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=cross_validation, scoring='roc_auc')
    grid_search.fit(x_train_df['text'], y_train_bow)
    
    best_model = grid_search.best_estimator_
    print(f"Best AUROC: {grid_search.best_score_}")
    print(f"Best parameters: {grid_search.best_params_}")

    y_proba_test = best_model.predict_proba(x_test_df['text'])[:, 1]

    np.savetxt('yproba1_test.txt', y_proba_test)

if __name__ == '__main__':
    main()
