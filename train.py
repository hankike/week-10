import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor


def roast_category(roast):
    roast_mapping = {
        # Map each of our roasts with a number
        # This must be done before get_coffee_data as that function uses this one
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }

    return roast_mapping.get(roast)


def get_coffee_data():
    # getting our data from the link
    df_coffee = pd.read_csv(
        "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

    # Grabbing our categories
    df_coffee.dropna(subset=['100g_USD', 'rating'])

    # Using apply
    df_coffee['roast_cat'] = df_coffee['roast'].apply(roast_category)

    return df_coffee

# Linear regression


def train_linear_regression(df_coffee):
    X = df_coffee[['100g_USD']]
    y = df_coffee['rating']

    model = LogisticRegression()
    model.fit(X, y)

    return model

# Decision tree


def train_decision_tree(df_coffee):
    X = df_coffee[['100g_USD', 'roast_cat']]
    y = df_coffee['rating']

    model = DecisionTreeRegressor()
    model.fit(X, y)

    return model


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    df_coffee = get_coffee_data()

    # For linear
    linear_model = train_linear_regression(df_coffee)
    save_model(linear_model, 'model_1.pickle')

    # For decision tree
    decision_tree_model = train_decision_tree(df_coffee)
    save_model(decision_tree_model, 'model_2.pickle')


if __name__ == "__main__":
    main()
