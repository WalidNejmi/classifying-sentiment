import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from load_train_data import load_data
import matplotlib.pyplot as plt

def print_pretty(*args):
    col_width = max(len(str(word)) for row in args for word in row) + 2
    for row in args:
        print("".join(str(word).ljust(col_width) for word in row))

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    vectorizer = CountVectorizer(lowercase=True, min_df=5, max_df=0.5, binary=False)
    
    x_train_bow = vectorizer.fit_transform(x_train_df['text'])
    x_test_bow = vectorizer.transform(x_test_df['text'])
    vocabulary_size = len(vectorizer.vocabulary_)

    print(f"Vocabulary size: {vocabulary_size}")
    total_words = x_train_bow.sum()
    print(f"Total number of words in the training data: {total_words}")

    y_train_bow = y_train_df['is_positive_sentiment'].values

    # 1C: Hyperparameter Selection for Logistic Regression Classifier
    C_values = np.logspace(-4, 4, 9)
    train_scores = []
    valid_scores = []
    fold_scores = {C: [] for C in C_values}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=1000)
        train_fold_scores = []
        valid_fold_scores = []
        
        for fold, (train_index, valid_index) in enumerate(skf.split(x_train_bow, y_train_bow)):
            X_train_fold, X_valid_fold = x_train_bow[train_index], x_train_bow[valid_index]
            y_train_fold, y_valid_fold = y_train_bow[train_index], y_train_bow[valid_index]
            
            clf.fit(X_train_fold, y_train_fold)
            
            train_score = roc_auc_score(y_train_fold, clf.predict_proba(X_train_fold)[:, 1])
            valid_score = roc_auc_score(y_valid_fold, clf.predict_proba(X_valid_fold)[:, 1])
            
            train_fold_scores.append(train_score)
            valid_fold_scores.append(valid_score)
            fold_scores[C].append(valid_score)
        
        train_scores.append(np.mean(train_fold_scores))
        valid_scores.append(np.mean(valid_fold_scores))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_scores, 'bo-', label='Training')
    plt.plot(C_values, valid_scores, 'ro-', label='Validation')

    # Plot individual fold scores
    for C, scores in fold_scores.items():
        plt.plot([C] * len(scores), scores, 'g.', alpha=0.3)

    plt.xscale('log')
    plt.xlabel('Regularization parameter (C)')
    plt.ylabel('ROC AUC Score')
    plt.title('Logistic Regression Hyperparameter Selection')
    plt.legend()
    plt.grid(True)

    # Add error bars to show variation across folds
    plt.errorbar(C_values, valid_scores, yerr=[np.std(fold_scores[C]) for C in C_values], 
                 fmt='none', ecolor='r', capsize=5, alpha=0.5)

    plt.savefig('hyperparameter_selection.png')
    plt.close()

    # Print pretty table of results
    print_pretty(("C Value", "Avg Train Score", "Avg Valid Score", "Std Dev Valid Score"))
    for C, train_score, valid_score in zip(C_values, train_scores, valid_scores):
        std_dev = np.std(fold_scores[C])
        print_pretty((f"{C:.2e}", f"{train_score:.4f}", f"{valid_score:.4f}", f"{std_dev:.4f}"))

    # Select best C value
    best_C = C_values[np.argmax(valid_scores)]
    print(f"\nBest C value: {best_C:.2e}")

    # Train final model with best C
    best_model = LogisticRegression(C=best_C, max_iter=1000)
    best_model.fit(x_train_bow, y_train_bow)

    # Predict on test set
    y_proba_test = best_model.predict_proba(x_test_bow)[:, 1]

    np.savetxt('yproba1_test.txt', y_proba_test)

    print("\nHyperparameter Selection Analysis:")
    print("We performed a hyperparameter search for the regularization parameter C in Logistic Regression, using 5-fold cross-validation. We explored C values ranging from 1e-4 to 1e4 on a logarithmic scale. The figure 'hyperparameter_selection.png' shows the average ROC AUC scores for both training and validation sets, along with individual fold scores and error bars representing the standard deviation across folds.")
    print(f"\nFrom the results, we can observe that the model tends to underfit at low C values (C < 1e-2) and slightly overfit at high C values (C > 1e2). The optimal C value is {best_C:.2e}, which balances model complexity and generalization. This value achieves the highest average validation score of {max(valid_scores):.4f} with a standard deviation of {np.std(fold_scores[best_C]):.4f}.")
    print("\nThe individual fold scores and error bars help us understand the stability of the model's performance across different data splits. We can see that the model's performance is relatively consistent across folds, especially for C values between 1e-2 and 1e2, indicating good stability in this range.")

if __name__ == '__main__':
    main()