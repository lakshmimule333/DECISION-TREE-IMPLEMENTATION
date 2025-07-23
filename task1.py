# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("📊 Iris Dataset Preview:")
print(X.head())

# Split the dataset (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Train Decision Tree Classifier
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# Predict
y_pred = dtree.predict(X_test)

# Evaluate
print("\n✅ Model Evaluation:")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Sample Predictions
print("\n🔍 Sample Predictions:")
sample_df = pd.DataFrame({
    'Actual': [iris.target_names[val] for val in y_test[:10]],
    'Predicted': [iris.target_names[val] for val in y_pred[:10]]
})
print(sample_df)

# Plot 1: Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    dtree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("🌳 Decision Tree (max_depth=3)")
plt.tight_layout()
plt.show()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("🔹 Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# Plot 3: Feature Importances
importances = pd.Series(dtree.feature_importances_, index=iris.feature_names)
plt.figure(figsize=(8, 4))
importances.sort_values().plot(kind='barh', color='teal')
plt.title("📈 Feature Importances from Decision Tree")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
