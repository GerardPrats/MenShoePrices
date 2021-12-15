import pandas as pd
import numpy as np


# Removing redundant columns for our analysis:
def remove_columns(df, columnnames):
    df.drop([c for c in columnnames], axis=1, inplace=True)

    # Show processed data without unnecessary columns:
    print("Total entries: ", df.shape[0], "; total fields: ", df.shape[1])
    pd.set_option("display.max_columns", None)
    print(df.info())
    return df


# Functions to replace the values of columns:
def colors_tf(x):
    if x is np.nan:
        return 0
    return len(x.split(","))


def asins_tf(x):
    if x is np.nan:
        return False
    return True

def manu_tf(x):
    if x is np.nan:
        return "No Manufacturer"
    return x


# Transform all data:
def transform_data(df):
    # Transform data where NaN can be interpreted as something.
    df = df[df["brand"].notnull()]
    df["brand"] = df["brand"].apply(lambda x: x.lower())

    df["colors"] = df["colors"].apply(colors_tf)
    df["datechanged"] = df["dateadded"] != df["dateupdated"]

    df.drop(["dateadded", "dateupdated", "features", "prices_condition", "merchants", "categories"], axis=1,
            inplace=True)
    df["descriptions"] = pd.notnull(df["descriptions"])
    df["reviews"] = pd.notnull(df["reviews"])

    # Remove rows with NaN issale, prices_amountmin:
    for c in ["prices_issale", "prices_amountmin"]:
        df = df[df[c].notnull()]

    # After removing null values, transform prices_amount to numeric:
    df["prices_amountmin"] = pd.to_numeric(df["prices_amountmin"], errors="coerce", downcast="float")
    df["prices_amountmax"] = pd.to_numeric(df["prices_amountmax"], errors="coerce", downcast="float")

    df["price"] = (df["prices_amountmin"] + df["prices_amountmax"]) / 2
    df.drop(["prices_amountmin", "prices_amountmax"], axis=1, inplace=True)

    # Analyze the range of prices to get rid of outliers:
    print("Lowest price: ", df["price"].min())
    print("Biggest price: ", df["price"].max())

    # A price of 0.0 doesn't make sense, we can get rid of those entries:
    df = df[df["price"] > 0.0]

    df = df[df["prices_currency"] == "USD"]
    df["amazon"] = df["asins"].apply(asins_tf)
    df.drop(["asins"], axis=1, inplace=True)
    df["manufacturernumber"] = df["manufacturernumber"].apply(manu_tf)

    # Reset the indices after removing rows:
    df = df.reset_index(drop=True)
    print("Total entries: ", df.shape[0], "; total fields: ", df.shape[1])
    print(df.info())
    return df
