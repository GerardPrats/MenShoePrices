import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def show_data():
    # Loading the database into Python.
    df = pd.read_csv("./data/train.csv", low_memory=False)

    # Show general information about the database.
    pd.set_option("display.max_columns", None)
    print("Total entries: ", df.shape[0], "; total features: ", df.shape[1])
    df.head(5)
    return df


def check_null(df):
    # Check which columns have null values:
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])


# Transform issale to bool:
def sale_tf(x):
    if x == "true":
        return True
    return False


# Compare data to reviews:
def compare_reviews(df):
    # Changing attributes:
    df["brand_id"] = pd.factorize(df["brand"])[0] + 1
    df["manufacturer_id"] = pd.factorize(df["manufacturernumber"])[0] + 1
    df["prices_issale"] = df["prices_issale"].apply(sale_tf)

    # Picking only the most popular brands:
    df_filter = df[df["brand"].isin(df["brand"].value_counts()[df["brand"].value_counts() > 40].index)]

    # Calculating the correlation and showing the data:
    corr = df_filter[["brand_id", "manufacturer_id", "colors", "descriptions", "prices_issale", "reviews", "datechanged", "price", "amazon"]].corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    return df_filter


# Check individual attributes, grouping by brand:
def groupby_brand(df):
    # Generating the data for Nike:
    df_nike = df[df["brand"] == "nike"]

    corr = df_nike[["manufacturer_id", "colors", "descriptions", "prices_issale", "reviews", "datechanged", "price",
                    "amazon"]].corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    # Generating the data for Nike:
    df_ralph = df[df["brand"] == "ralph lauren"]

    corr = df_ralph[["manufacturer_id", "colors", "descriptions", "prices_issale", "reviews", "datechanged", "price",
                     "amazon"]].corr()
    sns.heatmap(corr, annot=True)
    plt.show()
