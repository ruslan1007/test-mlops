import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def run():
    
    iris = load_iris()

    X, y = iris['data'].tolist(), iris['target'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.9)

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=predictions)
    print(acc)

if __name__ == "__main__":
    run()