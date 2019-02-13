import json
import pandas as pd
import numpy as np


# Filters all the files with a certain attribute to the same file.
def filter_json_by_attribute(filename, attribute, value, output_path):
    output_file = open(output_path, 'w')  # open the output file.
    with open(filename) as fp:
        for line in fp:
            obj = json.loads(line)
            obj_value = str(obj[attribute])
            if obj_value == value:
                output_file.write(json.dumps(obj) + '\n')
    output_file.close()


# Loads a .json file to pandas data frame.
def json_2_dataframe(filename):
    return pd.read_json(path_or_buf=filename, lines='true')


# Loads an attribute and concatenates into a dataframe.
def load_attribute(df, filename, id_attribute, attribute, default_value=0):
    # Loads the file that contains the attribute.
    df_attribute = pd.read_json(path_or_buf=filename)
    # Iterates over the ids present in both dataframes. This code also set the null cells to a default value.
    return df.reset_index().merge(df_attribute[[id_attribute, attribute]], left_index=True, left_on=id_attribute,
                    right_on=id_attribute, how='left').fillna({attribute: default_value}).set_index('index')

