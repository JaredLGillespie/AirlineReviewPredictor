import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, average_precision_score,precision_recall_curve, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import stop_words


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
X_train, X_test, y_train, y_test = train_test_split(data['content'],
                                                    data['recommended'],
                                                    random_state=1)

# Vectorize
# N-gram and Bigram
ngram_size = 2
vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size),
                             # the analyzer determines how you split up your ngram, i think 'word' is default?
                             # 'char' would be the character ngrams you were talking about
                             analyzer='word',
                             stop_words=stop_words.ENGLISH_STOP_WORDS)
vectorizer.fit(X_train)
train_ngram = vectorizer.transform(X_train)
test_ngram = vectorizer.transform(X_test)
# Naive Bayes
naive_bayes = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
naive_bayes.fit(train_ngram, y_train)

# predict the validation set
predictions = naive_bayes.predict(test_ngram)


# Validate
print('Accuracy: ' + format(accuracy_score(y_test, predictions)))
print('Precision: ' + format(precision_score(y_test, predictions)))
print('Recall: ' + format(recall_score(y_test, predictions)))
print('Confusion:\n' + format(confusion_matrix(y_test, predictions)))

# Done
