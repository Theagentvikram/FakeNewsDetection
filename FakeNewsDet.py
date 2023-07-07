import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the fake news CSV file
fake_news_df = pd.read_csv(r"C:\Users\abhic\Downloads\Fake.csv")

# Read the real news CSV file
real_news_df = pd.read_csv(r"C:\Users\abhic\Downloads\True.csv")

# Assign labels: 0 for fake news, 1 for real news
fake_news_df['label'] = 0
real_news_df['label'] = 1

# Combine the dataframes into a single dataframe
news_df = pd.concat([fake_news_df, real_news_df], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(news_df['text'], news_df['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

# Predict the labels for the testing data
y_pred = logreg.predict(X_test_tfidf)

#Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', confusion_mat)
print('Classification Report:\n', classification_rep)
