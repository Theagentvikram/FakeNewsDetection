import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')
print(stopwords.words('english'))

df = pd.read_csv(r"C:\Users\abhic\Downloads\faketest.csv")
df.shape
df.head()
# looking and replacing null datas
df.isnull().sum()
df = df.fillna('')
# merging author name and news title for easier use
df['content'] = df['title']+' '+df['text']+df['date']
print(df['content'])
print("wait..")
# Stemming is the process of reducing a word to Root word
ps= PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
df['content'] = df['content'].apply(stemming)
x = df['content'].values
y = df['label'].values
# converting the textual to numerical data for easier use
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify=y, random_state=2)
model = LogisticRegression()
model.fit(x_train,y_train)
x_train_pred = model.predict(x_train)
training_data_acc = accuracy_score(x_train_pred,y_train)
print(f"Accuracy score: {training_data_acc}")
x_test_pred = model.predict(x_test)
test_data_acc = accuracy_score(x_test_pred,y_test)
print(f"Accuracy score: {test_data_acc}")
x_new = x_test[1]
print (f"NB TEST ",x_new)
prediction = model.predict(x_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')