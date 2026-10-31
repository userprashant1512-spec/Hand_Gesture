
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Load collected dataset
DATA_FILE = "gesture_data.csv"

df = pd.read_csv(DATA_FILE)
df = df.dropna()

print("Total samples:", len(df))
print("\nSamples per gesture:")
print(df["label"].value_counts().sort_index())

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"].astype(str)

# Check that every gesture has enough samples
if y.nunique() < 2:
    raise ValueError("Collect data for at least two different gestures.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each gesture needs at least 2 samples. Collect more data."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the ML model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

#  Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Save the trained model
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved as gesture_model.pkl")
print("Training completed successfully!")
