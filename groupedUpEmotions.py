

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_parquet('C:/Users/alexa/Desktop/CODE/3.Semester/AI - module/GoEmotionNewFinal/goEmotionsFinal.parquet')

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
# Define emotion groups
emotion_groups = {
    'admiration': 'positive', 'amusement': 'positive', 'approval': 'positive', 'caring': 'positive', 
    'curiosity': 'positive', 'excitement': 'positive', 'gratitude': 'positive', 'joy': 'positive', 
    'love': 'positive', 'optimism': 'positive', 'relief': 'positive', 'desire': 'positive', 

    'anger': 'negative', 'annoyance': 'negative', 'disappointment': 'negative', 'disapproval': 'negative', 
    'disgust': 'negative', 'embarrassment': 'negative', 'fear': 'negative', 'grief': 'negative', 
    'remorse': 'negative', 'sadness': 'negative', 'nervousness': 'negative', 

    'confusion': 'ambiguous', 'surprise': 'ambiguous', 'realization': 'ambiguous', 'pride': 'ambiguous', 
    'neutral': 'ambiguous'
}
data['emotion'] = data['label'].map(emotion_map)
data['emotion_group'] = data['emotion'].map(emotion_groups)

# Prepare data for training
X = data['text']
y_group = data['emotion_group']
label_binarizer = LabelBinarizer()
y_group_onehot = label_binarizer.fit_transform(y_group)

X_train, X_test, y_train, y_test = train_test_split(X, y_group_onehot, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Define and compile the model
model = Sequential([
    Dense(128, input_shape=(5000,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train, epochs=10, batch_size=32, validation_data=(X_test_vectorized, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_vectorized, y_test)
print(f"Overall Test Accuracy: {test_accuracy:.2f}")

# Predict and calculate accuracy for each group
y_pred = np.argmax(model.predict(X_test_vectorized), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
group_labels = label_binarizer.classes_
predicted_groups = [group_labels[i] for i in y_pred]
true_groups = [group_labels[i] for i in y_test_labels]

from sklearn.metrics import accuracy_score
group_accuracies = {}
for group in ["positive", "negative", "ambiguous"]:
    group_indices = [i for i, true_label in enumerate(true_groups) if true_label == group]
    group_accuracies[group] = accuracy_score(
        [true_groups[i] for i in group_indices],
        [predicted_groups[i] for i in group_indices]
    )
    print(f"Accuracy for {group}: {group_accuracies[group]:.2f}")
