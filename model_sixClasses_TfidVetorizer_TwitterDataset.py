import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_parquet('C:/Users/alexa/Desktop/CODE/3.Semester/AI - module/ProjectTwitterSentimentAnalysis/Other Vectorizer/dataTwitter.parquet')

# Map the labels to emotion names for readability (optional)
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['emotion'] = data['label'].map(emotion_map)

# Splitting data
X = data['text']  # Feature: tweet text
y = data['label']  # Target: emotion labels (0-5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Encode the labels as one-hot for neural network
y_train_onehot = to_categorical(y_train, num_classes=6)
y_test_onehot = to_categorical(y_test, num_classes=6)

# Define the neural network model
model = Sequential([
    Dense(128, input_shape=(5000,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train_onehot, epochs=1, batch_size=32, validation_data=(X_test_vectorized, y_test_onehot))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_vectorized, y_test_onehot)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate a classification report
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(X_test_vectorized), axis=1)
print(classification_report(y_test, y_pred, target_names=emotion_map.values()))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
 #Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



# Define emotion labels for the confusion matrix plot
emotion_labels = list(emotion_map.values())

# Generate predictions
y_pred = model.predict(X_test_vectorized)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.show()

# (Optional) Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_classes, target_names=emotion_map.values()))

# Calculate per-class accuracy
accuracies = {}
for i, emotion in emotion_map.items():
    # Extract the correct predictions for each emotion
    correct_preds = conf_matrix[i, i]
    total_preds = conf_matrix[i].sum()
    accuracies[emotion] = correct_preds / total_preds if total_preds > 0 else 0

# Print per-class accuracy to console
for emotion, accuracy in accuracies.items():
    print(f"Accuracy for {emotion}: {accuracy:.2f}")

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.title('Per-Class Accuracy')
plt.xlabel('Emotion')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Accuracy range from 0 to 1
plt.show()

from sklearn.metrics import precision_recall_fscore_support

# Calculate precision, recall, and F1 score for each class
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes)

# Plot precision, recall, and F1 score
metrics_df = pd.DataFrame({
    'Emotion': emotion_labels,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

metrics_df.plot(x='Emotion', kind='bar', figsize=(12, 6), color=['#4CAF50', '#2196F3', '#FF5722'])
plt.title('Precision, Recall, and F1-Score per Class')
plt.xlabel('Emotion')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='upper right')
plt.show()

# Calculate the maximum probability for each prediction
max_probs = np.max(y_pred, axis=1)
correct = max_probs[y_pred_classes == y_test]
incorrect = max_probs[y_pred_classes != y_test]


# Calculate text lengths
data['text_length'] = data['text'].apply(len)

# Plot text length distribution by emotion
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='emotion', y='text_length', order=emotion_labels)
plt.title('Text Length Distribution by Emotion')
plt.xlabel('Emotion')
plt.ylabel('Text Length')
plt.show()

#  pass in a text input, process it using the same tokenizer and padding setup
#  used during training, and then
#  call model.predict to get the predicted probabilities for each emotion class.
def predict_emotion(text, model, tokenizer, max_length, emotion_map):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict the emotion
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map the predicted class to the emotion name
    emotion = emotion_map[predicted_class]
    confidence = prediction[0][predicted_class]
    
    print(f"Input Text: {text}")
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
    return emotion

# Example usage
text_input = "I am so happy to be part of this project!"
predicted_emotion = predict_emotion(text_input, model, tokenizer, max_length, emotion_map)