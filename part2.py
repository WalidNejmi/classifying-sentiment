import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_train_data import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

class NegationHandler:
    def __init__(self, negation_words=None, window_size=5):
        self.negation_words = negation_words or {"not", "no", "never", "without"}
        self.window_size = window_size

    def transform(self, texts):
        return [self._handle_negation(text) for text in texts]

    def _handle_negation(self, text):
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if words[i].lower() in self.negation_words:
                for j in range(i + 1, min(i + self.window_size, len(words))):
                    if words[j].lower() in self.negation_words:
                        result.append("positive")
                        i = j + 1
                        break
                else:
                    result.append(words[i])
                    i += 1
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)

def create_model(input_dim, units=128, dropout_rate=0.5, learning_rate=0.001):
    model = Sequential([
        Dense(units, activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    x_train_df, y_train_df, x_test_df = load_data()

    # Apply negation handling
    negation_handler = NegationHandler()
    x_train_transformed = negation_handler.transform(x_train_df['text'])
    x_test_transformed = negation_handler.transform(x_test_df['text'])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
    x_train_tfidf = vectorizer.fit_transform(x_train_transformed)
    x_test_tfidf = vectorizer.transform(x_test_transformed)

    # Standardization
    scaler = StandardScaler(with_mean=False)
    x_train_scaled = scaler.fit_transform(x_train_tfidf)
    x_test_scaled = scaler.transform(x_test_tfidf)

    # Split the training data for validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_scaled, y_train_df['is_positive_sentiment'], test_size=0.2, random_state=42
    )

    # Define the input dimension for the model
    input_dim = x_train.shape[1]

    # Use KerasClassifier with the model creation function
    model = KerasClassifier(build_fn=create_model, input_dim=input_dim, verbose=0)

    # Define the parameter space
    param_distributions = {
        'epochs': [50, 100],
        'batch_size': [32, 64, 128],
        'units': [32, 64, 128],
        'dropout_rate': [0.3, 0.5, 0.7],
        'learning_rate': [0.001, 0.01]
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                                       n_iter=20, cv=3, verbose=1, scoring='roc_auc', n_jobs=-1)
    
    random_search.fit(x_train, y_train)

    print(f"Best AUROC: {random_search.best_score_}")
    print(f"Best parameters: {random_search.best_params_}")

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_proba_test = best_model.predict_proba(x_test_scaled)[:, 1]

    # Save predictions to file
    np.savetxt('yproba2_test.txt', y_proba_test)

if __name__ == '__main__':
    main()

