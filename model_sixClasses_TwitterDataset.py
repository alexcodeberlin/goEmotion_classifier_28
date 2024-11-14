# Import necessary libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
data = pd.read_parquet('C:/Users/alexa/Desktop/CODE/3.Semester/AI - module/ProjectTwitterSentimentAnalysis/dataTwitter.parquet')

# Map labels to emotion names for readability (optional)
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['emotion'] = data['label'].map(emotion_map)

# Separate features and target
X = data['text']  # Text data
y = data['label']  # Emotion labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing: Tokenizing and padding sequences
vocab_size = 5000  # Max number of words to consider in vocabulary
embedding_dim = 64  # Dimension of word embeddings
max_length = 100  # Max length for sequences
oov_token = "<OOV>"  # Token for out-of-vocabulary words

# Initialize tokenizer and fit on training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences and pad them to ensure equal length
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Define the neural network model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # Output layer for 6 emotion classes
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 10  # Number of training epochs
history = model.fit(X_train_padded, y_train, epochs=epochs, validation_data=(X_test_padded, y_test), verbose=2)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=2)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate predictions and print a classification report
y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=emotion_map.values()))

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


# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

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

# Assuming df is a DataFrame with columns: 'Emotion', 'Topic', 'Frequency'
# Generate a pivot table for the heatmap
pivot_table = df.pivot(index="Emotion", columns="Topic", values="Frequency")

# Plot the heatmap
plt.figure(figsize=(13, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 8})

# Customize plot appearance
plt.xticks(rotation=45, ha="right")
plt.title('Emotion-Topic Correlation Heatmap', fontsize=16)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Emotion', fontsize=12)

plt.tight_layout()
plt.show()

# Save the model if desired
#model.save('/kaggle/working/emotion_classification_model.h5')
