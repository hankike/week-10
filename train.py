import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


def get_coffee_data():
    # getting our data from the link
    df_coffee = pd.read_csv(
        "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    df_coffee.dropna(subset=['100g_USD', 'rating'])

    return df_coffee


def train_model(df_coffee):
    X = df_coffee[['100g_USD']]
    y = df_coffee['rating']

    model = LogisticRegression()
    model.fit(X, y)

    return model


def save_model(model):
    with open('model_1.pickle', 'wb') as file:
        pickle.dump(model, file)


def main():
    df_coffee = get_coffee_data()
    model = train_model(df_coffee)
    save_model(model)


if __name__ == "__main__":
    main()
