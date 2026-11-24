# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.
2.Split Dataset into Training and Testing Sets.
3.Train the Model Using Stochastic Gradient Descent (SGD).
4.Make Predictions and Evaluate Accuracy.
5.Generate Confusion Matrix.


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by:RANTHIYA S
RegisterNumber:212225230225  
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.head()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
Head:
<img width="827" height="232" alt="image" src="https://github.com/user-attachments/assets/a99c89ce-6e47-4482-8731-65820c97ed26" />

Accuracy:
<img width="195" height="29" alt="image" src="https://github.com/user-attachments/assets/a27200c1-5458-4fb9-b44e-9782ed9995ce" />

Confusion Matrix:
<img width="210" height="110" alt="image" src="https://github.com/user-attachments/assets/623270a4-189c-4cb9-810b-a58bed7a7827" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
