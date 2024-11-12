import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_parquet('C:/Users/alexa/Desktop/CODE/3.Semester/AI - module/GoEmotionNewFinal/goEmotionsFinal.parquet')

# Map the labels to emotion names for readability
emotion_map = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

data['emotion'] = data['label'].map(emotion_map)

# Splitting data
X = data['text']  # Feature: tweet text
y = data['label']  # Target: emotion labels (0-27)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Encode the labels as one-hot for neural network
y_train_onehot = to_categorical(y_train, num_classes=28)
y_test_onehot = to_categorical(y_test, num_classes=28)

# Define the neural network model
model = Sequential([
    Dense(128, input_shape=(5000,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(28, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train_onehot, epochs=10, batch_size=32, validation_data=(X_test_vectorized, y_test_onehot))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_vectorized, y_test_onehot)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate a classification report
y_pred = np.argmax(model.predict(X_test_vectorized), axis=1)
print(classification_report(y_test, y_pred, target_names=emotion_map.values()))

# Calculate precision, recall, and F1 score for each class
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

# Print per-class accuracy to console
for i, emotion in emotion_map.items():
    print(f"{emotion}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1 Score={f1[i]:.2f}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = np.arange(len(emotion_map))

# Create bar plots for precision, recall, and F1 score
plt.bar(index, precision, bar_width, alpha=0.6, label='Precision')
plt.bar(index + bar_width, recall, bar_width, alpha=0.6, label='Recall')
plt.bar(index + 2 * bar_width, f1, bar_width, alpha=0.6, label='F1 Score')

plt.title('Precision, Recall, and F1 Score per Emotion')
plt.xticks(index + bar_width, emotion_map.values(), rotation=45)
plt.ylabel('Scores')
plt.legend()
plt.tight_layout()
plt.show()

# Function to predict emotion from text
def predict_emotion(text):
    text_vectorized = vectorizer.transform([text]).toarray()  # Transform the input text
    prediction = model.predict(text_vectorized)  # Make prediction
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class
    return emotion_map[predicted_class]  # Return the corresponding emotion

# Example usage of the predict_emotion function
input_text = "I'm feeling really happy today!"  # Replace with any input text
predicted_emotion = predict_emotion(input_text)
print(f"The predicted emotion for the input text is: {predicted_emotion}")

# Count the occurrences of each class in the test set
class_counts = pd.Series(y_test).value_counts()

# 1. Identify the most frequent class and its count
most_frequent_class = class_counts.idxmax()  # Class with the maximum count
most_frequent_count = class_counts.max()      # Count of the most frequent class

# 2. Calculate accuracy by predicting the most frequent class
total_samples = len(y_test)  # Total number of instances in the test set
max_class_accuracy = most_frequent_count / total_samples  # Calculate accuracy

# Print the results
print(f"The most frequent class is: {emotion_map[most_frequent_class]} with {most_frequent_count} instances.")
print(f"Accuracy by always predicting the most frequent class: {max_class_accuracy:.2f} or {max_class_accuracy * 100:.2f}%")
