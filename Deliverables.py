import pandas as pd

# Data Collection
df_true = pd.read_csv(r"S:\Dataset\True.csv")
df_fake = pd.read_csv(r"S:\Dataset\Fake.csv")

# Add labels to the datasets
df_true['label'] = 0  # Real news
df_fake['label'] = 1  # Fake news

# Data Preparation
df = pd.concat([df_true, df_fake], ignore_index=True)

# Choose the model (Logistic Regression as an example)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_train_pred = model.predict(X_train_vectorized)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test_vectorized)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Parameter Tuning (if required)

# Making Predictions
text = input("Enter the news text:")
text_vectorized = vectorizer.transform([text])
prediction = model.predict(text_vectorized)[0]

if prediction == 0:
    print("The news is Real")
else:
    print("The news is Fake")

# Print the accuracies
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
