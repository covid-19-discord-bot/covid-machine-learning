# coding=utf-8
# noinspection SpellCheckingInspection
"""
All credits to Murtaza's Workshop
https://www.youtube.com/watch?v=6CZiz-FLZF0
I simply made PyCharm happier by removing typos
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filename='spam.log',
        filemode='w'
    )


logger = logging.getLogger("ml_models-predict")


def predict_model(name: str = "OWID_WRL", key: str = "total_cases"):
    # load data
    logger.info("Loading CSV...")
    data = pd.read_csv(f"../data_sources/{'world' if name == 'OWID_WRL' else name.upper()}_data.csv")
    data = data[["index", key]]

    # prepare data
    logger.info("Preparing data...")
    x = np.array(data[key]).reshape(-1, 1)
    y = np.array(data["index"]).reshape(-1, 1)

    logger.info("Finding best feature...")
    max_accuracy = [0, 0]
    for i in range(2, 10):
        # prepare PolynomialFeature
        poly_feature = PolynomialFeatures(degree=i)
        a = poly_feature.fit_transform(x)

        # training data
        model = LinearRegression()
        model.fit(a, y)
        ac = model.score(a, y)
        if ac > max_accuracy[1]:
            max_accuracy[0] = i
            max_accuracy[1] = ac
        logger.debug(f"Accuracy is {ac} on degree {i}")

    # prepare PolynomialFeature
    logger.info("Preparing best feature...")
    poly_feature = PolynomialFeatures(degree=max_accuracy[0])
    x = poly_feature.fit_transform(x)

    # training data
    logger.info("Training model...")
    model = LinearRegression()
    model.fit(x, y)

    # prediction
    logger.info("Predicting...")
    days = 28  # number of days to predict
    final_day = len(data["index"])  # last day in the dataset
    # print(f"Cases after {days} days: {int(model.predict(poly_feature.fit_transform([[final_day + days]])))}")

    # return
    x1 = np.array(list(range(1, final_day + days))).reshape(-1, 1)
    y1 = model.predict(poly_feature.fit_transform(x1))
    return y1


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    predict_model()
