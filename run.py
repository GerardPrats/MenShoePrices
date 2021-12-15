from src import showing_data
from src import process_data
from src import model_training

if __name__ == "__main__":
    # Show the data and the null values we have in our dataset:
    df = showing_data.show_data()
    showing_data.check_null(df)
    print("")

    # Remove unnecessary columns:
    df = process_data.remove_columns(df, ["id", "count", "dimension", "ean", "flavors", "imageurls", "isbn", "manufacturer",
                             "prices_availability", "prices_color", "prices_count", "prices_flavor",
                             "prices_merchant", "prices_offer", "prices_returnpolicy", "prices_size", "prices_shipping",
                             "prices_source", "prices_sourceurls", "prices_warranty", "quantities", "skus", "sourceurls",
                             "upc", "vin", "websiteids", "weight", "keys", "prices_dateadded",
                             "prices_dateseen", "sizes", "name"])
    print("")
    df = process_data.transform_data(df)

    df_filter = showing_data.compare_reviews(df)
    showing_data.groupby_brand(df_filter)
    model_training.split_menshoe(df_filter)