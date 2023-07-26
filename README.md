# Titanic Kaggle Competition - README

This repository contains code for participating in the Titanic competition on Kaggle. The objective of the competition is to predict whether passengers aboard the Titanic survived or not, based on various features such as age, sex, ticket class, and more.

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib

You can install them using pip:

```
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository to your local machine:

```
git clone https://github.com/your-username/titanic-kaggle.git
cd titanic-kaggle
```

2. Download the Titanic dataset (`train.csv`) from Kaggle or provide the path to the file in the code where it reads the dataset.

```python
df = pd.read_csv("/Users/pb/Downloads/titanic/train.csv")
```

3. Data Preprocessing

   - The 'Sex' column is encoded using LabelEncoder to convert categorical data to numerical values (0 for one category and 1 for the other).

   - Missing values in the "Age" column are replaced with the mean age of the dataset.

   - Unnecessary columns like "Embarked", "PassengerId", "Name", "Ticket", and "Cabin" are dropped from the features.

4. Split the dataset into training and test sets using `train_test_split`.

5. Model Selection and Training

   - The model chosen for this task is a Random Forest Classifier with 100 estimators.

   - The model is trained on the training data using the `fit` method.

6. Model Evaluation

   - The accuracy of the model is computed on the test set using `accuracy_score`.

   - The "out-of-bag" (oob) score of the Random Forest Classifier is also displayed.

   - ROC-AUC score, precision, recall, and F1-score are computed using various evaluation metrics from scikit-learn.

7. Visualization

   - Precision-Recall Curve is plotted to visualize the precision and recall trade-off.

   - Precision vs. Recall plot is displayed to explore the relationship between precision and recall.

   - ROC Curve is plotted to visualize the true positive rate (sensitivity) against the false positive rate (1-specificity).

8. Running the Code

   To run the code, ensure you have the required libraries installed and have the Titanic dataset in the correct path or adjust the path in the `pd.read_csv()` function accordingly. Then, simply execute the code in your Python environment.

```
python titanic.py
```

## Plots

### Precision and Recall Plot
![Precision-Recall Curve](https://github.com/Piyush-Bhor/titanic-kaggle/blob/main/plots/1.png)

### Precision vs. Recall Plot
![PrecisionVsRecall Curve](https://github.com/Piyush-Bhor/titanic-kaggle/blob/main/plots/2.png)

### ROC Curve
![ROC Curve](https://github.com/Piyush-Bhor/titanic-kaggle/blob/main/plots/3.png)

## Disclaimer

Keep in mind that this is a basic implementation, and there are many ways to improve the model's performance, such as hyperparameter tuning, feature engineering, or using different machine learning algorithms. This code serves as a starting point for your exploration in the Titanic Kaggle competition.

Feel free to explore, modify, and experiment with the code to enhance your results.

Happy coding and good luck with the competition!
