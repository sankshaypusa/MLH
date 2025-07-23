# MLH

🫀 Project Description: EF Prediction using VideoMAE, BART, and ReLU Regressor (with RAG-style fine-tuning)
This project focuses on predicting Ejection Fraction (EF) — a critical clinical measure of cardiac function — from echocardiographic videos. The EF indicates how well the heart's left ventricle pumps blood with each beat and is vital for diagnosing heart failure, cardiomyopathy, and other cardiovascular conditions. Our method combines state-of-the-art video transformer models, semantic language encoders, and retrieval-augmented learning techniques to enable accurate EF regression from raw ultrasound video data.

The core idea is to extract rich spatiotemporal embeddings using VideoMAE (a masked autoencoder for video data), enrich these embeddings using a frozen BART encoder, and train a multi-layer ReLU-based regressor to predict the EF. A Retrieval-Augmented Generation (RAG) style approach is employed, where for every test video, its most similar training samples are retrieved using cosine similarity, and the regressor is locally fine-tuned using only those neighbors. This makes the model personalized, efficient, and more robust to small datasets.

🔄 Step-by-Step Procedure
🔹 Step 1: Dataset Preparation
Begin with the EchoNet-Dynamic dataset, which contains thousands of echocardiographic videos labeled with EF values. You should organize the dataset as follows:

A folder Videos/ that holds .avi format videos.

A CSV file FileList.csv containing three columns:

FileName: The base name of each video file.

EF: The EF label (as a float).

Split: Whether the video is used for TRAIN, VAL, or TEST.

This CSV helps manage your training and evaluation splits while preserving metadata.

🔹 Step 2: Video Embedding Using VideoMAE
Each video is processed using VideoMAE, a pretrained transformer model designed to understand temporal patterns in video data. For each video:

Randomly sample 16 evenly spaced frames using a helper function.

Convert frames to RGB format using PyAV.

Pass the frames through the videomae-base-finetuned-kinetics model.

Extract the final hidden state from the model's output (a 768-dimensional vector).

Save the resulting embedding as a .npy file inside the Embeddings_BART/ directory.

This process is repeated for all videos in the TRAIN and VAL sets.

🔹 Step 3: Load and Normalize the Data
After generating embeddings, load them into memory and separate the dataset into training and test splits. Normalize the EF labels to the range [0, 1] based on the min and max values observed in the training and validation sets. Normalization improves the training stability of the neural network regressor and ensures consistent convergence.

🔹 Step 4: Define the EF Prediction Model
The prediction model consists of:

A frozen BART encoder from the facebook/bart-base model. It maps the 768-dimensional embedding into a deeper semantic space using the [CLS] token from its final encoder layer.

A projection layer that aligns the raw video embedding to the BART embedding space.

A multi-layer ReLU regressor that takes the [CLS] token and predicts a scalar EF value (initially normalized).

This hybrid architecture enables the use of pretrained language models to interpret dense, high-dimensional video embeddings in a semantic way.

🔹 Step 5: Retrieval-Augmented Learning (RAG-style Fine-tuning)
This is the key innovation of the pipeline. For every test video:

Compute cosine similarity between its embedding and all training embeddings.

Retrieve the top-30 most similar training samples.

Train the regressor on this subset for 300 epochs, allowing the model to specialize locally.

Use the trained model to predict EF for the test video.

Denormalize the predicted EF back to the original scale.

This RAG-like fine-tuning approach ensures that the model adapts to the most relevant training data per test case, leading to better generalization and reduced overfitting.

🔹 Step 6: Evaluation
Once all test EF values have been predicted, compute the Mean Absolute Error (MAE) between predicted and actual EF scores. The MAE provides a straightforward metric for evaluating regression performance. In your implementation, this final MAE is printed out at the end of the script and serves as the primary performance indicator.

✅ Summary of Key Features
Model-Free Embedding: Uses pretrained VideoMAE for zero-shot video understanding.

Transformer Fusion: Enhances embeddings using BART’s semantic encoder.

Lightweight Regressor: Simple yet powerful MLP with ReLU and dropout layers.

Instance-Level Fine-tuning: Uses top-30 nearest neighbors for each test case.

Plug-and-Play: Easily extendable to new video data with minimal retraining.

📈 Output Example
The script prints progress during embedding generation and per-sample prediction. At the end, it displays the final MAE, such as:
✅ Final MAE (VideoMAE + BART encoder + ReLU regressor): 7.12
If you'd like, I can save this as a .md file or provide additional sections like setup instructions for Colab, GPU acceleration, or API inference wrapper.
