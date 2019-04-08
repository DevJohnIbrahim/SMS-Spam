import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def GNBtest (X_test):
    return clf.predict(X_test)


def GNBtrain (X_train,y_train):
    return clf.fit(X_train, y_train)


def GNBAccuracy (y_test,pred):
    return accuracy_score(y_test,pred)


data = pd.read_csv('spam.csv',encoding='latin1')


print(" ")
print(data.head())

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
print(" ")
print(data.head())

y = data['v1'].as_matrix()
X_text = data['v2'].as_matrix()
print(" ")
print(X_text.shape)
print(y.shape)

sw = stopwords.words("english")
#cv = CountVectorizer(stop_words =sw)
#tcv = cv.fit_transform(X_text).toarray()
#print(" ")
#print(len(tcv[0,:]))
#print(tcv.shape)
#print(y.shape)

vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True)
X = vectorizer.fit_transform(X_text).toarray()
print(" ")
print(X.shape)
print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.202, random_state=42)
print(" ")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


clf = GaussianNB()
print("Training The Naive bayes classifier")
print(" ")
GNBtrain (X_train,y_train)
print("Testing The Naive bayes classifier")
print(" ")
predic = GNBtest(X_test)
print("Calculating accuracy")
print(" ")
accuracy = GNBAccuracy(y_test,predic)

print("Accuracy:",accuracy*100)
