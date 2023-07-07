import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the fake news dataset
df_fake = pd.read_csv(r"C:\Users\abhic\Downloads\Fake.csv")
df_fake['label'] = 1  # Assign label 1 for fake news

# Load the true news dataset
df_true = pd.read_csv(r"C:\Users\abhic\Downloads\True.csv")
df_true['label'] = 0  # Assign label 0 for true news

# Concatenate the two datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Preprocess the data
x = df['text']  # Features: Textual data
y = df['label']  # Target: Labels (0 for true, 1 for fake)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test_vec)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
