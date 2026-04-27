# ============================================
# Cross-Domain Sentiment Analysis
# Train: Amazon + IMDB
# Test : Yelp
# Model: BERT + LSTM
# Classes: Negative, Neutral, Positive
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, TFBertModel


# --------------------------------------------
# 1️⃣ Detect Dataset Folder
# --------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(base_path, "Datasets")

print("\nBase Path:", base_path)
print("Dataset Folder:", dataset_folder)
print("Files inside Datasets folder:", os.listdir(dataset_folder))


def find_file(keyword):
    for file in os.listdir(dataset_folder):
        if keyword.lower() in file.lower():
            return os.path.join(dataset_folder, file)
    raise FileNotFoundError(f"No file containing '{keyword}' found!")


amazon_path = find_file("amazon")
imdb_path = find_file("imdb")
yelp_path = find_file("yelp")

print("\nDetected Files:")
print("Amazon:", amazon_path)
print("IMDB  :", imdb_path)
print("Yelp  :", yelp_path)


# --------------------------------------------
# 2️⃣ Load Dataset (.txt format)
# --------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["text", "label"]
    return df


amazon_df = load_data(amazon_path)
imdb_df = load_data(imdb_path)
yelp_df = load_data(yelp_path)


# --------------------------------------------
# 3️⃣ Train / Test Setup
# --------------------------------------------
train_df = pd.concat([amazon_df, imdb_df], ignore_index=True)
test_df = yelp_df

print("\nTraining size:", len(train_df))
print("Testing size :", len(test_df))


# --------------------------------------------
# 4️⃣ Tokenization
# --------------------------------------------
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )


train_encodings = encode_texts(train_df["text"])
test_encodings = encode_texts(test_df["text"])

train_labels = tf.convert_to_tensor(train_df["label"].values)
test_labels = tf.convert_to_tensor(test_df["label"].values)


# --------------------------------------------
# Save Tokenizer (.pkl)
# --------------------------------------------
with open("bert_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved as bert_tokenizer.pkl")


# --------------------------------------------
# Save Encoded Dataset (.npy)
# --------------------------------------------
np.save("train_input_ids.npy", train_encodings["input_ids"].numpy())
np.save("train_attention_mask.npy", train_encodings["attention_mask"].numpy())

np.save("test_input_ids.npy", test_encodings["input_ids"].numpy())
np.save("test_attention_mask.npy", test_encodings["attention_mask"].numpy())

print("Encoded datasets saved as .npy files")


# --------------------------------------------
# TensorFlow Dataset
# --------------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"]
    },
    train_labels
)).shuffle(1000).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    },
    test_labels
)).batch(BATCH_SIZE)


# --------------------------------------------
# 5️⃣ Build BERT + LSTM Model
# --------------------------------------------
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

bert_output = bert_model(input_ids, attention_mask=attention_mask)

sequence_output = bert_output.last_hidden_state

# LSTM Layer
lstm = tf.keras.layers.LSTM(64)(sequence_output)

# Dropout
dropout = tf.keras.layers.Dropout(0.3)(lstm)

# Softmax Output
output = tf.keras.layers.Dense(3, activation="softmax")(dropout)

model = tf.keras.Model(
    inputs=[input_ids, attention_mask],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------
# 6️⃣ Train Model
# --------------------------------------------
print("\nTraining Started...\n")

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)


# --------------------------------------------
# Save Model (.h5)
# --------------------------------------------
model.save("bert_lstm_sentiment_model.h5")

print("Model saved as bert_lstm_sentiment_model.h5")


# --------------------------------------------
# 7️⃣ Evaluate Model
# --------------------------------------------
print("\nEvaluating on Yelp (Unseen Domain)...\n")

predictions = model.predict(test_dataset)

pred_labels = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_df["label"], pred_labels)

print("Cross-Domain Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(test_df["label"], pred_labels))


# --------------------------------------------
# 8️⃣ Sentiment Label Mapping
# --------------------------------------------
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# --------------------------------------------
# 9️⃣ Print First 5 Predictions
# --------------------------------------------
print("\nFirst 5 Yelp Test Predictions:\n")

for i in range(5):
    print("Text      :", test_df["text"].iloc[i])
    print("Actual    :", label_map[test_df["label"].iloc[i]])
    print("Predicted :", label_map[pred_labels[i]])
    print("---------------------------------------------------")
    # --------------------------------------------
# Save Results for Evaluation
# --------------------------------------------
np.save("y_true.npy", test_labels)
np.save("y_pred.npy", pred_labels)
np.save("y_pred_prob.npy", predictions)

print("Saved prediction files for evaluation!")