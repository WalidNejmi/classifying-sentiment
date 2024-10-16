import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from load_train_data import load_data
from scipy.stats import uniform, randint
from sklearn.base import BaseEstimator, TransformerMixin
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def analyze_predictions(model, X, y, texts, n_examples=7):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    false_positives = np.where((y_pred == 1) & (y == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y == 1))[0]
    
    fp_indices = false_positives[np.argsort(y_proba[false_positives])[-n_examples:]]
    fn_indices = false_negatives[np.argsort(y_proba[false_negatives])[:n_examples]]
    
    return fp_indices, fn_indices, y_proba[fp_indices], y_proba[fn_indices]

def plot_examples(texts, indices, probas, title):
    plt.figure(figsize=(12, len(indices) * 1.2))
    for i, (idx, proba) in enumerate(zip(indices, probas)):
        text = texts[idx]
        plt.text(0.05, 1 - (i + 0.5) / len(indices), f"{text}\n(Probability: {proba:.3f})", 
                 fontsize=10, wrap=True, va='center')
    plt.yticks([])
    plt.xticks([])
    plt.title(title)
    plt.tight_layout()

class NegationHandler(BaseEstimator, TransformerMixin):
    def __init__(self, negation_words=None, window_size=3):
        self.negation_words = negation_words or [
            'not', 'no', 'never', "n't", 'without', 'hardly', 'barely',
            'scarcely', 'seldom', 'rarely', 'neither', 'nor'
        ]
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._handle_negation(text) for text in X]

    def _handle_negation(self, text):
        words = text.split()
        result = []
        negation_positions = []
        for i, word in enumerate(words):
            if word.lower() in self.negation_words:
                negation_positions.append(i)
            
            if len(negation_positions) >= 2 and i - negation_positions[-2] <= self.window_size:
                result.append('POSITIVE')
                negation_positions = []
            else:
                result.append(word)
        
        return ' '.join(result)

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    # Create a pipeline for feature extraction and classification
    pipeline = Pipeline([
        ('negation', NegationHandler()),
        ('tfidf', TfidfVectorizer(lowercase=True, binary=True)),
        ('classifier', SVC(random_state=42, probability=True, class_weight='balanced'))
    ])

    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        'tfidf__max_df': uniform(0.5, 0.5),
        'tfidf__min_df': randint(1, 20),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__C': np.logspace(-5, 5, 25),
        'classifier__kernel': ['linear', 'rbf'], 
        'classifier__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.999).rvs(10)),
    }

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # Perform randomized search
    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=cv, 
                                       scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(x_train_df['text'], y_train_df['is_positive_sentiment'])

    print(f"Best AUROC: {random_search.best_score_}")
    print(f"Best parameters: {random_search.best_params_}")

    # Get the best model
    best_model = random_search.best_estimator_

    # Analyze predictions on a validation set
    train_index, val_index = next(cv.split(x_train_df['text'], y_train_df['is_positive_sentiment']))
    X_val = x_train_df['text'].iloc[val_index]
    y_val = y_train_df['is_positive_sentiment'].iloc[val_index]
    texts_val = X_val.values

    y_pred_val = best_model.predict(X_val)
    y_proba_val = best_model.predict_proba(X_val)[:, 1]

    # Analyze false positives and false negatives
    fp_indices = np.where((y_pred_val == 1) & (y_val == 0))[0]
    fn_indices = np.where((y_pred_val == 0) & (y_val == 1))[0]

    # Calculate AUROC for false positives and false negatives
    if len(fp_indices) > 0:
        auroc_fp = roc_auc_score(y_val[y_pred_val == 1], y_proba_val[y_pred_val == 1])
        print(f"AUROC for false positives: {auroc_fp:.4f}")
    else:
        print("No false positives found in the validation set.")

    if len(fn_indices) > 0:
        auroc_fn = roc_auc_score(y_val[y_pred_val == 0], y_proba_val[y_pred_val == 0])
        print(f"AUROC for false negatives: {auroc_fn:.4f}")
    else:
        print("No false negatives found in the validation set.")

    # Plot examples of false positives and false negatives
    n_examples = 7
    fp_indices = fp_indices[np.argsort(y_proba_val[fp_indices])[-n_examples:]]
    fn_indices = fn_indices[np.argsort(y_proba_val[fn_indices])[:n_examples]]

    plot_examples(texts_val, fp_indices, y_proba_val[fp_indices], "False Positives")
    plt.savefig('false_positives_part2.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_examples(texts_val, fn_indices, y_proba_val[fn_indices], "False Negatives")
    plt.savefig('false_negatives_part2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Make predictions on the test set
    y_proba_test = best_model.predict_proba(x_test_df['text'])[:, 1]

    # Save predictions to file
    np.savetxt('yproba2_test.txt', y_proba_test)

if __name__ == '__main__':
    main()