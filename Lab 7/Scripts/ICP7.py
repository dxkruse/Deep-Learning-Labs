from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import nltk

import requests
from bs4 import BeautifulSoup
import os

#%%
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

#tfidf_Vect = TfidfVectorizer()
tfidf_Vect = TfidfVectorizer(stop_words='english')
#tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
#clf = SVC()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

#%%

# Initialize URL
url = "https://en.wikipedia.org/wiki/Google"

# Save website contents to html variable
html = requests.get(url)

# Parse all html content into bsObj variable
bsObj = BeautifulSoup(html.content, "html.parser")
body = bsObj.find('div', {'class': 'mw-parser-output'})
# Print title of page
print(bsObj.title.string)

# Set file name and open file
file_name = "input.txt"
file = open('input.txt','a+',encoding='utf-8')
data = str(body.text)    
# Iterate through all links and write each one to the desired file, adding a new
# line after each link.
file.write(str(body.text))
# Close the file
file.close()

#%%

wtokens = nltk.word_tokenize(data)
stokens = nltk.sent_tokenize(data)
pos = nltk.pos_tag(wtokens)

pStemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stems = list()
lems = list()
trigrams = list()
for i in range(len(wtokens)):
    stems.append(pStemmer.stem(wtokens[i]))
    lems.append(lemmatizer.lemmatize(wtokens[i]))
    
ner = nltk.ne_chunk(nltk.pos_tag(nltk.wordpunct_tokenize(data)))

for i in range(len(stokens)):
    trigrams.append(nltk.trigrams(stokens[i]))