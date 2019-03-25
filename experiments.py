from os import listdir

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Instantiate the parser
parser = argparse.ArgumentParser(description='Map segmentation tool.', formatter_class=argparse.RawTextHelpFormatter)

# Required input_file name argument.
parser.add_argument('--input_dirs', required=False, nargs='+', default=[],
                    help='Strings representing dirs with analysis input files.')

args = parser.parse_args()
files = listdir(args.input_dirs[0])
dfs = [None] * len(files)
# Loads all the input files.
for index, input_file in enumerate(files):
    df = pd.read_csv(args.input_dirs[0] + '/' + input_file)
    df['confs'] = [str(x) for x in range(len(df))]
    dfs[index] = df


for index, df in enumerate(dfs):
    plt.plot(df['confs'], df['result'], marker='o', label=files[index])
plt.legend()
plt.xlabel("confs")
plt.ylabel("result")
plt.show()
