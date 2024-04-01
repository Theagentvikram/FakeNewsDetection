import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib  # Use joblib for model persistence
import json

# Load your labeled dataset (replace with your actual data)
data = [
    {"text": "Working on a project", "category": "Work"},
    {"text": "Reading a book", "category": "Personal"},
    {"text": "Going to the gym", "category": "Health"},
    {"text": "Meeting with friends", "category": "Social"},
    {"text": "Watching a movie", "category": "Entertainment"},
    # Add more examples...
]

df = pd.DataFrame(data)

# Split the data into training and testing sets
train_data = df

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(train_data['text'], train_data['category'])

# Save the trained model to a file
model_filename = "activity_model.joblib"
joblib.dump(model, model_filename)

# Load user-provided dataset (replace with your own data handling logic)
data_filename = r"S:\test.json"
try:
    with open(data_filename, "r") as f:
        user_data = json.load(f)
except FileNotFoundError:
    user_data = {}

# Main Streamlit app
st.title("Daily Activity Assistant")

# User input
user_input = st.text_input("Tell me about your activity:")

if user_input:
    # Make predictions for user input
    predicted_category = model.predict([user_input])[0]

    st.write(f"Predicted Category: {predicted_category}")

    # Add logic to provide assistance or information based on the predicted category
    if predicted_category == 'Work':
        st.write("You're working on a project. Remember to stay focused and take breaks.")
    elif predicted_category == 'Personal':
        st.write("Enjoy your personal time! Reading a book is a great choice.")
    elif predicted_category == 'Health':
        st.write("Taking care of your health is important. Going to the gym is a good habit.")
    elif predicted_category == 'Social':
        st.write("Meeting with friends is a great way to socialize and have a good time.")
    elif predicted_category == 'Entertainment':
        st.write("Watching a movie can be a fun and relaxing activity.")

    # Update daily activities
    st.header("Update Daily Activities")

    activity_name = st.text_input("Activity Name:")
    activity_category = st.selectbox("Category:", ["Work", "Personal", "Health", "Social", "Entertainment"])
    activity_time = st.time_input("Time:")
    activity_day = st.date_input("Date:")

    new_activity = {
        "name": activity_name,
        "category": activity_category,
        "time": activity_time.strftime("%H:%M"),
        "day": str(activity_day),
    }

    if st.button("Add Activity"):
        # Update activities
        if activity_category not in user_data:
            user_data[activity_category] = []

        user_data[activity_category].append(new_activity)

        # Save the updated data to the file
        with open(data_filename, "w") as f:
            json.dump(user_data, f, indent=2)

        st.success("Activity added successfully!")

# Note: For a production scenario, consider more robust error handling and user feedback.
