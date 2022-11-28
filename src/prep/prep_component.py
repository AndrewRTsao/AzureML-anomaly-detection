import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Columns from input train dataset that will be used by the model to determine if there is an outlier, e.g. price vs sales (case-sensitive)
outlier_columns = ["Item_MRP", "Item_Outlet_Sales"] # Include actual column names as strings in the outlier_columns list


def parse_args():

    # setting up argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--raw_data", type=str, help="Path to raw data file"
    )
    parser.add_argument(
        "--prep_data", type=str, help="Path of prepared training data"
    )

    # parse and return args
    args = parser.parse_args()
    return args


def convert_column_to_numerical(df, column):

    unique_values = np.unique(df[column])

    i = 1
    for unique_value in unique_values:
        print("Unique value being updated in column")
        print(unique_value)  
        df.loc[df[column] == unique_value, column] = i
        i = i+1

    return df


def prepare_data(raw_data, prep_data):

    # Retrieving training file from earlier created ABS bucket and container (uploaded via Kaggle dataset), plus creating the dataframe
    with open(raw_data, "r") as handle:
        input_df = pd.read_csv(Path(raw_data))

    # Prune training dataset to only include columns needed for anomaly detection model and convert any strings / categoricals into numeric (for pyod / numpy)
    output_df = pd.DataFrame() # Construct empty df that you will then append to

    for column in outlier_columns:
        print("Checking if column is numerical")
        if input_df[column].dtype != np.number:
            
            print("Column is not numerical so updating it")    
            input_df = convert_column_to_numerical(input_df, column)

        # Appending updated column to output df
        data_to_append = input_df[column]
        output_df = pd.concat([output_df, data_to_append], axis=1)

    # Writing output dataframe
    print("writing output")
    output_df.to_csv((Path(prep_data) / "training_prepared.csv"))


def main(args):

    prepare_data(args.raw_data, args.prep_data)


if __name__ == "__main__":

    # parse args and pass it to the main function
    args = parse_args()
    main(args)
