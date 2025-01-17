import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import train_test_split
def makePrediction(modelName):
    MarketData = pd.read_csv('financialMarketData.csv')
    model = joblib.load(modelName)
    print("model successfully loaded")
    X = MarketData.drop(columns=['Y', 'Data'])
    y = MarketData['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    modelStats = classification_report(y_test, y_pred, output_dict=True)
    print(modelStats)

    return modelStats