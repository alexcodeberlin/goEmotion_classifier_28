# Emotion Classification Model

This project builds a neural network to classify text into emotional categories using the **GoEmotions dataset**. It provides two classification levels:
1. **28 Distinct Emotions**: Classifies data into individual emotions:admiration, amusement, anger, annoyance, approval,caring,confusion,curiosity,desire,disappointment","disapproval","disgust","embarrassment","excitement","fear",gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral
3. **3 Grouped Categories**: Groups the 28 emotions into **positive**, **negative**, and **ambiguous** categories for broader sentiment analysis.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Acknowledgments](#acknowledgments)

## Overview

This project uses TensorFlow and Keras to build a neural network model for emotion classification. Data processing and feature extraction are managed using Pandas and Scikit-Learn. The model is trained on labeled data from the GoEmotions dataset, which provides 28 distinct emotions, and a grouped version that condenses these emotions into three broader categories: **positive**, **negative**, and **ambiguous**.


## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Manually install the necessary packages:

    ```bash
    pip install numpy pandas scikit-learn tensorflow
    ```

## Dataset Setup

Ensure your dataset file is available in the specified directory, with a path that matches the one used in your code:

- **Dataset Path** in the code to change: `C:/Users/alexa/Desktop/CODE/3.Semester/AI - module/GoEmotionNewFinal/goEmotionsFinal.parquet`

## Running the Project

1. Open Visual Studio Code and load the project folder.
2. Open `emotion_model.py` (or the main Python script).
3. Run the script by pressing `F5` or by selecting "Run Python File" in the Run menu.

## Expected Output

The model will:
- Train on the 28 emotion labels in the dataset.

- 
- Evaluate and print test accuracies for both the 28-label classification and the 3-category grouping.
