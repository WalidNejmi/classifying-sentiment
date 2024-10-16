import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from load_train_data import load_data

def analyze_predictions(model, X, y, texts, n_examples=7):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    false_positives = np.where((y_pred == 1) & (y == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y == 1))[0]
    
    fp_indices = false_positives[np.argsort(y_proba[false_positives])[-n_examples:]]
    fn_indices = false_negatives[np.argsort(y_proba[false_negatives])[:n_examples]]
    
    return fp_indices, fn_indices, y_proba[fp_indices], y_proba[fn_indices]

def plot_examples(texts, indices, probas, title):
    plt.figure(figsize=(8, len(indices) * 1))  # Increase height per entry for better spacing
    for i, (idx, proba) in enumerate(zip(indices, probas)):
        text = texts[idx]
        # Adjust the y position to space out the text more evenly
        plt.text(0.05, 1 - (i + 1) / (len(indices) + 1), f"{text} (Probability: {proba:.3f})", 
                 fontsize=9, wrap=True, va='top')
    
    plt.yticks([])
    plt.xticks([])
    
    # Set the title and tweak the layout to avoid excessive margins
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to prevent overlap with title
    plt.show()

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    vectorizer = CountVectorizer(lowercase=True, min_df=1, max_df=0.415, binary=False)
    
    x_train_bow = vectorizer.fit_transform(x_train_df['text'])
    x_test_bow = vectorizer.transform(x_test_df['text'])
    vocabulary_size = len(vectorizer.vocabulary_)

    print(f"Vocabulary size: {vocabulary_size}")
    total_words = x_train_bow.sum()
    print(f"Total number of words in the training data: {total_words}")

    y_train = y_train_df['is_positive_sentiment'].values

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(lowercase=True, min_df=1, max_df=0.415, binary=False)),
        ('classifier', LogisticRegression(C=2.154434690031882, solver='lbfgs', max_iter=5000, random_state=12))
    ])

    cross_validation = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
    train_index, val_index = next(cross_validation.split(x_train_df['text'], y_train))
    X_train, X_val = x_train_df['text'].iloc[train_index], x_train_df['text'].iloc[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]
    texts_val = X_val.values
    websites_val = x_train_df['website_name'].iloc[val_index]

    pipeline.fit(X_train, y_train)
    
    # Analyze predictions
    fp_indices, fn_indices, fp_probas, fn_probas = analyze_predictions(pipeline, X_val, y_val, texts_val)

    # Plot examples
    try:
        plot_examples(texts_val, fp_indices, fp_probas, "False Positives")
        plt.savefig('false_positives.png', dpi=300, bbox_inches='tight')
        plt.close()

        plot_examples(texts_val, fn_indices, fn_probas, "False Negatives")
        plt.savefig('false_negatives.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error occurred while plotting: {e}")

    # Analyze performance on different review sources
    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    
    websites = ['amazon', 'imdb', 'yelp']
    for website in websites:
        website_mask = websites_val == website
        if np.sum(website_mask) > 0:
            auroc_website = roc_auc_score(y_val[website_mask], y_proba_val[website_mask])
            print(f"AUROC for {website} reviews: {auroc_website:.4f}")

    # Final prediction on test set
    y_proba_test = pipeline.predict_proba(x_test_df['text'])[:, 1]
    np.savetxt('yproba1_test.txt', y_proba_test)

    # Additional analysis
    sentence_lengths = [len(text.split()) for text in texts_val]
    y_pred_val = pipeline.predict(X_val)
    correct_predictions = y_pred_val == y_val

    avg_length_correct = np.mean([length for length, correct in zip(sentence_lengths, correct_predictions) if correct])
    avg_length_incorrect = np.mean([length for length, correct in zip(sentence_lengths, correct_predictions) if not correct])

    print(f"Average sentence length for correct predictions: {avg_length_correct:.2f}")
    print(f"Average sentence length for incorrect predictions: {avg_length_incorrect:.2f}")

    # Analyze performance on sentences with/without negation
    negation_words = ['not', 'no', 'never', "n't", 'without']
    has_negation = np.array([any(word in text.lower() for word in negation_words) for text in texts_val])
    auroc_with_negation = roc_auc_score(y_val[has_negation], pipeline.predict_proba(X_val[has_negation])[:, 1])
    auroc_without_negation = roc_auc_score(y_val[~has_negation], pipeline.predict_proba(X_val[~has_negation])[:, 1])

    print(f"AUROC for sentences with negation: {auroc_with_negation:.4f}")
    print(f"AUROC for sentences without negation: {auroc_without_negation:.4f}")

if __name__ == '__main__':
    main()
