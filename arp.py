import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, brier_score_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from time import time


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
                             stop_words=stop_words.ENGLISH_STOP_WORDS)
vectorizer.fit(X_train)
train_ngram = vectorizer.transform(X_train)
# scaling the data achieves worse results, but maybe its the ethical thing to do in data science?
# train_ngram = StandardScaler(with_mean=False).fit_transform(train_ngram)
test_ngram = vectorizer.transform(X_test)
toc = time()
print('Vectorization Time: %.02fs' % (toc - tic))

# Naive Bayes
tic = time()
naive_bayes = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
naive_bayes.fit(train_ngram, y_train)
toc = time()
print('Training Time: %0.2fs' % (toc - tic))

# predict the validation set
tic = time()
predictions = naive_bayes.predict(test_ngram)
toc = time()
print('Prediction Time: %0.2fs' % (toc - tic))

predicted_probas = naive_bayes.predict_proba(test_ngram)


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

# ROC Curve
skplt.metrics.plot_roc_curve(y_test, predicted_probas)
plt.show()

# Precision-Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Done
