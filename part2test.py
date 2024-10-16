import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from load_train_data import load_data
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    # Split the training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_df['text'], y_train_df['is_positive_sentiment'], 
        test_size=0.2, random_state=42, stratify=y_train_df['is_positive_sentiment']
    )

    # Create a pipeline for feature extraction and classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_df=0.5, binary=False)),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True
        ))
    ])

    # Define parameter distributions for random search
    param_distributions = {
        'tfidf__max_df': uniform(0.5, 0.5),
        'tfidf__min_df': randint(1, 10),
        'classifier__n_estimators': randint(100, 1000),
        'classifier__max_depth': randint(10, 100),
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 10),
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_leaf_nodes': randint(10, 100),
        'classifier__min_impurity_decrease': uniform(0, 0.1),
        'classifier__ccp_alpha': uniform(0, 0.1)
    }

    # Perform random search
    best_score = 0
    best_params = None
    best_model = None

    for _ in range(100):  # Number of iterations
        # Sample random parameters
        params = {k: v.rvs() if hasattr(v, 'rvs') else np.random.choice(v) for k, v in param_distributions.items()}
        
        # Set the parameters and fit the model
        pipeline.set_params(**params)
        pipeline.fit(x_train, y_train)
        
        # Evaluate on validation set
        y_pred_proba = pipeline.predict_proba(x_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        
        # Update best model if score is improved
        if score > best_score:
            best_score = score
            best_params = params
            best_model = pipeline

    print(f"Best AUROC: {best_score}")
    print(f"Best parameters: {best_params}")

    # Make predictions on the test set
    y_proba_test = best_model.predict_proba(x_test_df['text'])[:, 1]

    # Save predictions to file
    np.savetxt('yproba2_test.txt', y_proba_test)

if __name__ == '__main__':
    main()