# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Create a small dataset
data = {
    'text': [
        "Patient: John Doe\nDate: 2023-01-15\nService: Consultation\nAmount: $100",
        "Invoice for services not provided\nAmount: $500",
        "Patient: Jane Smith\nDate: 2023-02-10\nService: Lab Test\nAmount: $150",
        "Missing details in the invoice",
        "Patient: Bob Johnson\nDate: 2023-03-05\nService: X-ray\nAmount: $200",
    ],
    'label': ['valid', 'invalid', 'valid', 'invalid', 'valid']
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data into feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Display classification report
print(classification_report(y_test, predictions))
