import json
import pandas as pd


# Filters all the files with a certain attribute to the same file.
def filter_json(filename, attribute, value, output_path):
    output_file = open(output_path, 'w')  # open the output file.
    with open(filename) as fp:
        for line in fp:
            obj = json.loads(line)
            obj_value = str(obj[attribute])
            if obj_value == value:
                output_file.write(json.dumps(obj) + '\n')
    output_file.close()


# Gives a summary of the data.
def number_of_entries(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Loads a .json file to pandas data frame.
def json_2_dataframe(filename):
    return pd.read_json(path_or_buf=filename, lines='true')
