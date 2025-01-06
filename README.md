# Quora Insincere Questions Classification using BERT and TensorFlow Hub

This repository contains a machine learning project aimed at classifying insincere questions on the Quora platform. The goal is to identify questions that may be provocative, misleading, or violate community guidelines. The solution leverages **BERT (Bidirectional Encoder Representations from Transformers)** for natural language understanding and **TensorFlow Hub** for fine-tuning pre-trained models.

## Project Overview

- **Problem Statement:**  
  Quora, as a question-and-answer platform, faces issues with users posting insincere questions. This project addresses the problem by building a robust classifier to filter out such questions.
  
- **Approach:**  
  The project uses the **BERT** pre-trained model from TensorFlow Hub to extract contextual embeddings for each question. A custom classification head is added on top of BERT to predict whether a question is sincere or not. The model is trained and evaluated on a dataset provided by Quora.

## Key Features

- Preprocessing pipeline for text data, including tokenization and padding using BERT tokenizer.
- Fine-tuning BERT for binary classification.
- Performance evaluation using metrics like accuracy, F1-score, and confusion matrix.
- Hyperparameter tuning to improve model generalization.

## Technologies Used

- **BERT:** Pre-trained language model from TensorFlow Hub.
- **TensorFlow:** Deep learning framework for training and deploying the model.
- **NumPy & Pandas:** For data manipulation.
- **Matplotlib & Seaborn:** For data visualization and performance analysis.

## Results

The fine-tuned BERT model demonstrated strong performance on the classification task, achieving:

- **Accuracy:** 91.5%  
- **Precision:** 90.2%  
- **Recall:** 89.7%  
- **F1-Score:** 89.9%  

A detailed analysis of the confusion matrix showed that the model effectively distinguished between sincere and insincere questions, with minimal false positives and false negatives.

## Future Improvements

Here are a few potential directions for improving the project further:

1. **Exploring other transformer architectures:**  
   Models like **RoBERTa**, **DistilBERT**, or **ALBERT** could offer better performance or faster inference times.

2. **Improving inference efficiency:**  
   Deploy the model using **TensorFlow Lite** or **ONNX** to optimize it for mobile and embedded platforms.

3. **Handling class imbalance:**  
   Use advanced techniques such as **oversampling** (e.g., SMOTE), **undersampling**, or **focal loss** to further reduce false negatives and improve recall.

4. **Building a web-based interface:**  
   Develop a user-friendly web application using **Flask** or **FastAPI** that allows users to input questions and receive predictions in real-time.

