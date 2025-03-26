import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit App Title
st.title("K-Nearest Neighbors (KNN) Classifier")

# Step 1: Business Understanding
st.write("### Step 1: Business Understanding")
logging.info("Starting K-Nearest Neighbors (KNN) implementation.")

# Step 2: Data Understanding
st.write("### Step 2: Data Understanding")
logging.info("Loading dataset...")

# Using the Penguins dataset as an alternative to Iris
df = sns.load_dataset("penguins").dropna()
st.write("Dataset Preview")
st.dataframe(df.head())

# Selecting features and target
selected_features = ["bill_length_mm", "bill_depth_mm"]
X = df[selected_features]
y = df["species"].astype("category").cat.codes  # Encoding categorical target

# Step 3: Data Preparation
st.write("### Step 3: Data Preparation")
logging.info("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: User Input for Model Hyperparameters
n_neighbors = st.slider("Select Number of Neighbors", 1, 20, 5)
weights = st.selectbox("Select Weights", ["uniform", "distance"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")

# Train KNN Model
st.write("### Step 4: Model Training and Hyperparameter Tuning")
logging.info(f"Training KNN model with n_neighbors={n_neighbors} and weights={weights}...")
knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
knn.fit(X_train, y_train)

# Step 5: Evaluation
st.write("### Step 5: Model Evaluation")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Step 6: Visualization
st.write("### Step 6: Visualization - Decision Boundary")
logging.info("Visualizing decision boundary...")

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("KNN Decision Boundary")
    st.pyplot(plt)

# Plot Decision Boundary
plot_decision_boundary(knn, X_scaled, y)

# Final Log Message
logging.info("KNN implementation completed successfully.")