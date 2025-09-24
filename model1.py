import os
import sys

print("--- DIAGNOSTIC INFORMATION ---")
print(f"Python Executable: {sys.executable}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
print("--- END DIAGNOSTIC INFORMATION ---\n")



#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Libraries and Load Data
# 
# First, we'll import the necessary libraries for data manipulation, modeling, and visualization. We will then load the `bank-additional-full.csv` dataset.
# 
# **Note on GPU Support (`cuml`):**
# To run the model on a GPU, you need an NVIDIA GPU and the RAPIDS `cuml` library. Installation can be complex. If you don't have this setup, you can simply run the CPU part of this notebook (using `scikit-learn`), which is the standard approach.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Scikit-learn for CPU-based CART model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# RAPIDS cuML for GPU-based CART model (optional)
# If you don't have cuML, this import will fail. You can comment it out.
try:
    import cudf
    import cuml
    from cuml.model_selection import train_test_split as cuml_train_test_split
    from cuml.tree import DecisionTreeClassifier as cuml_DecisionTreeClassifier
    from cuml.metrics import accuracy_score as cuml_accuracy_score
    print("RAPIDS cuML found. GPU functionality will be available.")
    gpu_enabled = True
except ImportError:
    print("RAPIDS cuML not found. GPU functionality will be disabled.")
    gpu_enabled = False

# Set plot style
plt.style.use('ggplot')

# Load the dataset
# The dataset uses semicolons as separators
df = pd.read_csv('/mnt/c/Users/sudee/Desktop/work/Enhancing-CART-with-a-K-Means-and-Random-Forest-Ensemble-for-Robust-Segmentation/bank+marketing/bank/bank-full.csv', sep=';')

print("Dataset loaded successfully!")
print(df.head())


# ## Step 2: Data Cleaning and Preprocessing
# 
# Here, we'll prepare the data for modeling. This involves:
# 1.  **Dropping the 'duration' column**: As the data description file (`bank-additional-names.txt`) warns, this feature is not known before a call is made and highly affects the output, making it a data leak. For a realistic predictive model, it should be removed.
# 2.  **Encoding the Target Variable**: We will convert the target variable `y` from "yes"/"no" to 1/0.
# 3.  **Handling Categorical Features**: We'll use one-hot encoding to convert categorical columns into a numerical format that the model can understand. The description file notes that missing values are coded as "unknown", which we will treat as a distinct category.
# 4.  **Splitting the Data**: We will split the data into training and testing sets.

# In[ ]:


print("Starting data cleaning and preprocessing...")

# Drop the 'duration' column as recommended for a realistic model
df_cleaned = df.drop('duration', axis=1)
print("Dropped 'duration' column.")

# Convert target variable 'y' to binary (0 for 'no', 1 for 'yes')
df_cleaned['y'] = df_cleaned['y'].map({'no': 0, 'yes': 1})

# Separate features (X) and target (y)
X = df_cleaned.drop('y', axis=1)
y = df_cleaned['y']

# One-hot encode all categorical features
# This will automatically handle columns with text data
X_encoded = pd.get_dummies(X, drop_first=True)
print("Categorical features have been one-hot encoded.")
print(f"Original number of features: {X.shape[1]}")
print(f"Number of features after encoding: {X_encoded.shape[1]}")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData has been split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# ## Step 3: Build and Train the CART Model (CPU Approach)
# 
# We will now build and train our Classification and Regression Tree (CART) model using `scikit-learn`. CART models are trained in a single pass and do not use epochs. Instead of a progress bar for epochs, we will time the training process and then evaluate its performance by looking at key metrics.
# 
# We'll look at:
# - **Accuracy**: The overall percentage of correct predictions.
# - **Classification Report**: Provides precision, recall, and F1-score for each class.
# - **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives.

# In[ ]:


print("--- Training CART Model on CPU ---")

# Initialize the Decision Tree Classifier (CART)
# We use a max_depth to prevent the tree from overfitting
cart_cpu = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model
print("Training the model...")
cart_cpu.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred_cpu = cart_cpu.predict(X_test)

# --- Evaluate the Model ---
print("\n--- CPU Model Evaluation ---")

# Calculate accuracy
accuracy_cpu = accuracy_score(y_test, y_pred_cpu)
print(f"Accuracy: {accuracy_cpu:.4f}")

# Display the classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cpu, target_names=['No', 'Yes']))

# Display the confusion matrix
print("Confusion Matrix:")
cm_cpu = confusion_matrix(y_test, y_pred_cpu)
sns.heatmap(cm_cpu, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CPU Model Confusion Matrix')
plt.show()


# ## Step 4: Build and Train the CART Model (GPU Approach with RAPIDS cuML)
# 
# This section is for users with an NVIDIA GPU and the RAPIDS cuML library. The process is very similar, but we use `cudf` for the DataFrame and `cuml` for the model. This can be significantly faster for large datasets.
# 
# **Note:** If you do not have a GPU setup, the code below will be skipped.

# In[ ]:


if gpu_enabled:
    print("--- Training CART Model on GPU ---")

    # Convert pandas DataFrames to cuDF DataFrames for GPU processing
    # The tqdm progress bar shows the status of the data transfer
    print("Transferring data to GPU...")
    X_encoded_gpu = cudf.from_pandas(X_encoded)
    y_gpu = cudf.from_pandas(y)
    print("Data transferred.")

    # Split data on the GPU
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = cuml_train_test_split(
        X_encoded_gpu, y_gpu, test_size=0.2, random_state=42, stratify=y_gpu
    )

    # Initialize the cuML Decision Tree Classifier
    cart_gpu = cuml_DecisionTreeClassifier(max_depth=5, random_state=42)

    # Train the model on the GPU
    print("Training the model on GPU...")
    cart_gpu.fit(X_train_gpu, y_train_gpu)
    print("GPU model training complete.")

    # Make predictions
    y_pred_gpu = cart_gpu.predict(X_test_gpu)

    # --- Evaluate the GPU Model ---
    print("\n--- GPU Model Evaluation ---")

    # Convert results back to CPU (pandas/numpy) for scikit-learn metrics
    y_test_cpu = y_test_gpu.to_pandas()
    y_pred_cpu_from_gpu = y_pred_gpu.to_pandas()

    # Calculate accuracy
    accuracy_gpu = accuracy_score(y_test_cpu, y_pred_cpu_from_gpu)
    print(f"Accuracy: {accuracy_gpu:.4f}")

    # Display the classification report
    print("\nClassification Report:")
    print(classification_report(y_test_cpu, y_pred_cpu_from_gpu, target_names=['No', 'Yes']))

    # Display the confusion matrix
    print("Confusion Matrix:")
    cm_gpu = confusion_matrix(y_test_cpu, y_pred_cpu_from_gpu)
    sns.heatmap(cm_gpu, annot=True, fmt='d', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('GPU Model Confusion Matrix')
    plt.show()
else:
    print("Skipping GPU training as RAPIDS cuML is not installed or configured.")

