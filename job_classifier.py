import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset
df = pd.read_csv("data.csv")  # Make sure your dataset has a 'Job Title' column

# Step 2: Preprocess and map titles to general roles
def map_to_general_role(title):
    title = title.lower()
    if "data analyst" in title:
        return "Data Analyst"
    elif "business analyst" in title:
        return "Business Analyst"
    elif "research analyst" in title:
        return "Research Analyst"
    elif "product analyst" in title:
        return "Product Analyst"
    elif "marketing analyst" in title:
        return "Marketing Analyst"
    elif "operations analyst" in title:
        return "Operations Analyst"
    elif "finance analyst" in title or "financial analyst" in title:
        return "Financial Analyst"
    elif "reporting analyst" in title:
        return "Reporting Analyst"
    elif "consultant" in title:
        return "Consultant"
    else:
        return "Other"

df['Role'] = df['Job Title'].apply(map_to_general_role)

# Step 3: Vectorize the job titles
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Job Title'])
y = df['Role']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict a sample job title
sample_title = ["Junior Data Analyst"]
sample_vector = vectorizer.transform(sample_title)
sample_prediction = clf.predict(sample_vector)
print("Sample prediction:", sample_prediction[0])
