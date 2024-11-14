# Emotion Classification Model

This project aims to develop a neural network model capable of detecting and identifying the emotional state behind a post. It is using the goEmotions dataset from google with 140000 posts and a Twitter dataset with 1.3 million posts. 
1. **28 Distinct Emotions**: Classifies data into individual emotions:admiration, amusement, anger, annoyance, approval,caring,confusion,curiosity,desire,disappointment","disapproval","disgust","embarrassment","excitement","fear",gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral
2. **3 Grouped Categories**: Groups the 28 emotions into **positive**, **negative**, and **ambiguous** categories for broader sentiment analysis.<br>
3 **6 emotions** Classifies data into 6 different emotions with the Twitter dataset. The dataset has 1.3 million data entries.<br>
4 **6 emotions with other vectorizer** Classifies data into 6 different emotions with the Twitter dataset. It uses a different vectorzer and shows better evaluation statistics.


## Overview

This project uses TensorFlow and Keras to build a neural network model for emotion classification.  The model is trained on labeled data from the GoEmotions dataset, which provides 28 distinct emotions and a grouped version that condenses these emotions into three broader categories: **positive**, **negative**, and **ambiguous**.


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


- For `28_emotional_states.py` and `grouped_up_version.py` **Dataset Path** in the code to change: `yourpath/goEmotionsFinal.parquet`
- unzip the twitter_dataset.zip
- For `model_six_classes_twitterDataset.py` and `model_six_classes_Tfid_vectorizer.py` **Dataset Path** in the code to change: `yourpath/goEmotionsFinal.parquet`


## Running the Project

1. Open Visual Studio Code and load the project folder.
2. Open `28_emotional_states.py`
3. Open `grouped_up_version.py`
4. Open `model_six_classes_twitterDataset.py`
5. Open `model_six_classes_Tfid_vectorizer.py`
6. Run the scripts 

## Expected Output

The model will:
- Train on the 28 emotion labels in the dataset.
- Train on a group up version with 28 emotion labels categorized into three groups: negative, positve and ambigous
- Evaluate and print test accuracies for both the 28-label classification and the 3-category grouping.
- plot F1 for every sentiment
