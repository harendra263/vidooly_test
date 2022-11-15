import pandas as pd
import numpy as np

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


SEED = 42

if __name__ == '__main__':
    scaled_df = pd.read_csv("input/scaled_df.csv")
    print(scaled_df.head())

    X = scaled_df.drop("adview", axis=1)
    y = scaled_df["adview"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    print(regressor.intercept_)

    print(regressor.coef_)

    y_pred = regressor.predict(X_test)

    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    print(df_preds)
