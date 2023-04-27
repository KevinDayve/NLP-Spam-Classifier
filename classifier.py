#Importing the necessary libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

texts  = pd.read_csv("SMSSpamCollection", sep = "\t", names = ["Label","Text"])
                     

#Initialising the WordNetLemmatizer
wordnet = WordNetLemmatizer()

#Initialising a list to store the import words weighted as per tf-idf method.
corpus= []
for i in range(len(texts)):
    replace = re.sub('[^a-zA-Z]', ' ', texts["Text"][i])
    replace = replace.lower()
    replace = replace.split()
    
    replace = [wordnet.lemmatize(word) for word in replace if word not in stopwords.words("english")]
    replace = " ".join(replace)
    corpus.append(replace)
                     
                     
#Importing Sklearn's TfIdf vector to convert words into vector array.
from sklearn.feature_extraction.text import TfidfVectorizer

tfvector = TfidfVectorizer()
X = tfvector.fit_transform(corpus).toarray()

#Creating dummy variables to make our inputs numeric

y = pd.get_dummies(texts["Label"])

#Turing the dependent variable into an array.
y = y.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size= 0.3, random_state = 101)

#Training the model using the naive bayes classifier

from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB()

spam_detection = bayes.fit(X_train, y_train)


#Prediction outcomes

pred = spam_detection.predict(X_test)

#Printing accuracy metrics
from sklearn.metrics import confusion_matrix, classification_report

confusion = confusion_matrix(y_test, pred)

#Printing the confusion matrix.
import seaborn as sns

print(confusion)

sns.heatmap(confusion, annot = True)

#Print the classification report
report = classification_report(y_test, pred)
print(report)