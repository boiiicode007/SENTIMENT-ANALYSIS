import pandas as pd
import numpy as np
import re # regular expression
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("review_dataset.csv")

# removing stopwords - words which do not make sense
# loading words in english
nltk.download( 'stopwords' )
sw = stopwords.words('english') 

def clean_text(sample):
    sample = sample.lower()
    sample = sample.replace("<br /><br />", " ")
    sample = re.sub("[^a-zA-Z]+", " ", sample)

    sample = sample.split(" ") # splitting the words
    sample = [ word for word in sample if word not in sw ]
    sample = " ".join(sample) # rejoining the words and creating the sentence

    return sample

df['review'] = df['review'].apply(clean_text) # each row will be passed to clean_text function

# X -> input
# y -> output
X = df['review'].values
y = df['label'].values

# training data
X_train = X[0:15000]
y_train = y[0:15000]
 
# test data
X_test = X[15000:20000]
y_test = y[15000:20000]

le = LabelEncoder()

# learning the mapping, pos:1, neg:0 ... it is just learning here
le.fit(y_train)

# converting pos/neg to 1/0
y_train = le.transform(y_train) # real transformation is happening here
y_test = le.transform(y_test)

# bag of word model
# converting words to number
cv = CountVectorizer(max_features=10000)

cv.fit(X_train) # learning the mapping
X_train = cv.transform(X_train).toarray() # transform is happening here, converting sparse to dense array using .toarray()

X_test = cv.transform(X_test).toarray()

mnb = MultinomialNB()

# testing part
mnb.fit(X_train, y_train)

mnb.predict(X_test)
print(mnb.score(X_test, y_test)) # checking how much perfect it is our accuracy

my_review = "the movie was bad"
my_review = clean_text(my_review)

my_review = cv.transform([my_review])
print(mnb.predict(my_review))