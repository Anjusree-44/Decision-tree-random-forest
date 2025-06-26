import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz
from sklearn import tree

# Load dataset
df = pd.read_csv(r"C:\Users\katta\OneDrive\Python-ai\myenv\heart.csv")

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title("Decision Tree")
plt.show()

# Accuracy and overfitting check
train_acc = dt.score(X_train, y_train)
test_acc = dt.score(X_test, y_test)
print(f"Decision Tree - Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")

# Cross-validation score
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
print(f"Decision Tree CV Score: {cv_scores_dt.mean():.3f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Accuracy and cross-validation
rf_acc = accuracy_score(y_test, rf_pred)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest - Accuracy: {rf_acc:.3f}, CV Score: {cv_scores_rf.mean():.3f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# Classification Report
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))
