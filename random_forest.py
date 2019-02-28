import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss

from sklearn.ensemble import RandomForestClassifier as Classifier

nltk.download('stopwords')

dataset = pd.read_csv('../../storage/dataset/classification.csv')
corpus = []

for summary in dataset['summary']:
    summary = summary.lower().split()

    ps = PorterStemmer()
    summary = [ps.stem(word) for word in summary if not word in set(stopwords.words('english'))]
    summary = ' '.join(summary)
    corpus.append(summary)

cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
total_log_loss = 0
total_accuracy = 0
best_accuracy = 0
best_report = None
best_type = ''
count = 0

for i in range(3, len(dataset.columns) - 1):
    y = dataset.iloc[:, i].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    # print("\n-------------------------------------------------\n")
    # print(dataset.columns[i] + ':')
    clf = Classifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    try:
        _log_loss = log_loss(y_test, y_pred)
        # print('Log Loss:')
        # print(_log_loss)

        # print('Accuracy Score:')
        accuracy = accuracy_score(y_test, y_pred)
        # print(accuracy * 100, '%')

        if accuracy < 0.9:
            total_accuracy += accuracy
            total_log_loss += _log_loss
            count += 1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_report = classification_report(y_test, y_pred)
                best_type = dataset.columns[i]

        # print('Classification Report:')
        # print(classification_report(y_test, y_pred))

        # print('Confusion Matrix:')
        # print(confusion_matrix(y_test, y_pred))
    except ValueError:
        print('')

# print("\n\n\n")
# print('Average Accuracy:')
# print((total_accuracy / count) * 100, '%')
# print('Average Log Loss:')
# print(total_log_loss / count)
print(best_type)
print(best_report)
print(best_accuracy)

