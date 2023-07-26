# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

# Importing the dataset
df = pd.read_csv("/Users/pb/Downloads/titanic/train.csv")

# Encoding column 'Sex'
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# Handling missing values in the "Age" column
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)

# Splitting into X and y
X = df.drop(["Survived", "Embarked", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)  # Features (all columns except "Survived")
y = df["Survived"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Random Forest model
model = RandomForestClassifier(n_estimators=100, oob_score = True)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(accuracy * 100) + "%")

model.score(X_train, y_train)
acc_random_forest = round(model.score(X_train, y_train) * 100, 2)
print("oob score:", round(model.oob_score_, 4)*100, "%")
print("acc_random_forest", acc_random_forest)

# Predict probabilities for the training data
y_scores = model.predict_proba(X_train)

# Access the predicted probabilities for the positive class
y_scores = y_scores[:, 1]

# ROC-AUC Curve
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)

# Confusion Matrix
predictions = cross_val_predict(model, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)

# Precision and Recall
print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))

# F1 Score
f1_score(y_train, predictions)

# Precision, Recall and Threshold Plot
precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

# Precison vs Recall Plot
def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()

# True positive rate vs False positive rate Plot
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
