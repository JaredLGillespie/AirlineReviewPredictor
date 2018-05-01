import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, brier_score_loss
from sklearn.naive_bayes import MultinomialNB
from time import time


def show_most_informative_features(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

    print('Top Features:')
    print('\tRecommended:')
    for coef, fn in coefs_with_fns[:-(n + 1): -1]:
        print('\t%.4f\t%s' % (coef, fn))

    print('\n\tNot Recommended:')
    for coef, fn in coefs_with_fns[:n]:
        print('\t%.4f\t%s' % (coef, fn))


# Datatypes of each column
dtype = {
    'airline_name': str,
    'link': str,
    'title': str,
    'author': str,
    'author_country': str,
    'date': str,
    'content': str,
    'aircraft': str,
    'type_traveller': str,
    'cabin_flown': str,
    'route': str,
    'overall_rating': float,
    'seat_comfort_rating': float,
    'cabin_staff_rating': float,
    'food_beverages_rating': float,
    'inflight_entertainment_rating': float,
    'ground_service_rating': float,
    'wifi_connectivity_rating': float,
    'value_money_rating': float,
    'recommended': int}

# Read in data
data = pd.read_table('airline.csv', sep=',', dtype=dtype)

# Remove unused columns
data = data.drop(['link', 'title', 'author', 'date'], axis=1)

# Describe data
print('Total: %s' % data.shape[0])
print('Not Recommended: %s' % data.where(data['recommended'] == 0).dropna(how='all').shape[0])
print('Recommended: %s' % data.where(data['recommended'] == 1).dropna(how='all').shape[0])

# ----- Single Validation & AdaBoost -----
# Split data
X_train, X_test, y_train, y_test = train_test_split(data['content'], data['recommended'], random_state=1)

# Vectorize
# N-gram and Bigram
ngram_size = 2
tic = time()
vectorizer = TfidfVectorizer(min_df=2,  # Remove words only appearing in a single document
                             ngram_range=(ngram_size, ngram_size),
                             # the analyzer determines how you split up your ngram, i think 'word' is default?
                             # 'char' would be the character ngrams you were talking about
                             analyzer='word',
                             stop_words=stop_words.ENGLISH_STOP_WORDS,
                             max_features=10000)
vectorizer.fit(X_train)
train_ngram = vectorizer.transform(X_train)
test_ngram = vectorizer.transform(X_test)
toc = time()
print('Vectorization Time: %.02fs' % (toc - tic))

# Important Features
print('Feature Count: %s' % len(vectorizer.get_feature_names()))

classifiers = [
    ['AdaBoost (Naive Bayes)', AdaBoostClassifier(base_estimator=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), n_estimators=100)],
    ['AdaBoost (Logistic Regression)', AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=100)],
    ['Naive Bayes', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)],
    ['Logistic Regression', LogisticRegression()]
]

for name, classifier in classifiers:
    print('%s%s%s' % ('-' * 20, name, '-' * 20))

    # Train
    tic = time()
    classifier.fit(train_ngram, y_train)
    toc = time()
    print('Training Time: %0.2fs' % (toc - tic))

    # Predict
    tic = time()
    predictions = classifier.predict(test_ngram)
    toc = time()
    print('Prediction Time: %0.2fs' % (toc - tic))

    predicted_probas = classifier.predict_proba(test_ngram)

    # Validate
    print('Accuracy: %0.3f' % accuracy_score(y_test, predictions))
    print('Precision: %0.3f' % precision_score(y_test, predictions))
    print('Recall: %0.3f' % recall_score(y_test, predictions))
    print('F1 Score: %0.3f' % f1_score(y_test, predictions))
    print('Brier Score: %0.3f' % brier_score_loss(y_test, predictions))
    print('Confusion:')
    conmat = np.array(confusion_matrix(y_test, predictions, labels=[1, 0]))
    confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                             columns=['predicted_positive', 'predicted_negative'])
    print(confusion)

    # AdaBoost doesn't give coefficients
    if 'AdaBoost' not in name:
        show_most_informative_features(vectorizer, classifier)

    # ROC Curve
    skplt.metrics.plot_roc_curve(y_test, predicted_probas)
    plt.savefig(name + 'roc curve.png')

    # Precision-Recall Curve
    skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
    plt.savefig(name + 'precision recall curve.png')

# ----- Cross Validation -----
# Split data
X, Y = data['content'], data['recommended']

# Vectorize
# N-gram and Bigram
ngram_size = 2
tic = time()
vectorizer = TfidfVectorizer(min_df=2,  # Remove words only appearing in a single document
                             ngram_range=(ngram_size, ngram_size),
                             # the analyzer determines how you split up your ngram, i think 'word' is default?
                             # 'char' would be the character ngrams you were talking about
                             analyzer='word',
                             stop_words=stop_words.ENGLISH_STOP_WORDS,
                             max_features=10000)
vectorizer.fit(X)
train_ngram = vectorizer.transform(X)
toc = time()
print('Vectorization Time: %.02fs' % (toc - tic))

# Important Features
print('Feature Count: %s' % len(vectorizer.get_feature_names()))

classifiers = [
    ['CrossVal AdaBoost (Naive Bayes)', AdaBoostClassifier(base_estimator=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), n_estimators=100)],
    ['CrossVal AdaBoost (Logistic Regression)', AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=100)],
    ['CrossVal Naive Bayes', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)],
    ['CrossVal Logistic Regression', LogisticRegression()]
]

for name, classifier in classifiers:
    print('%s%s%s' % ('-' * 20, name, '-' * 20))

    # Train & Predict
    tic = time()
    predictions = cross_val_predict(classifier, train_ngram, Y)
    toc = time()
    print('Cross Validation Time: %0.2f' % (toc - tic))

    # Validate
    print('Accuracy: %0.3f' % accuracy_score(Y, predictions))
    print('Precision: %0.3f' % precision_score(Y, predictions))
    print('Recall: %0.3f' % recall_score(Y, predictions))
    print('F1 Score: %0.3f' % f1_score(Y, predictions))
    print('Brier Score: %0.3f' % brier_score_loss(Y, predictions))
    print('Confusion:')
    conmat = np.array(confusion_matrix(Y, predictions, labels=[1, 0]))
    confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                             columns=['predicted_positive', 'predicted_negative'])
    print(confusion)

    # Probabilities
    predicted_probas = cross_val_predict(classifier, train_ngram, Y, method='predict_proba')

    # ROC Curve
    skplt.metrics.plot_roc_curve(Y, predicted_probas)
    plt.savefig(name + 'roc curve.png')

    # Precision-Recall Curve
    skplt.metrics.plot_precision_recall_curve(Y, predicted_probas)
    plt.savefig(name + 'precision recall curve.png')

# Done
