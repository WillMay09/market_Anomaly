import pandas as pd
import numpy as np
import seaborn as sns
import io
import base64
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

def getRegressionModelInfo():
    MarketData = pd.read_csv('financialMarketData.csv')
    numeric_columns = MarketData.select_dtypes(include=[np.number])

    heatMap_img = getCorrelationMatrix(numeric_columns)
    #get correlation matrix

    return heatMap_img
    

    
    

def getCorrelationMatrix(numeric_columns):
    corr_matrix = numeric_columns.corr()

    #Get the upper triangle of the correlation matrix to avoid duplicates
    upper_triangle = np.triu(corr_matrix)

    #Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)

    #Set up the matplotlib figure
    plt.figure(figsize=(12,8))

    #Create heatmap using seaborn
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)

    #Set the title of the plot
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    #create a io bytes object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
     #encode the image into base64 string for transfer
    return savePlotAsBase64(buf)
   
   
def savePlotAsBase64(buf):
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return img_base64