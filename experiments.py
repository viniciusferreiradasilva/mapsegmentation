import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Instantiate the parser
parser = argparse.ArgumentParser(description='Map segmentation tool.', formatter_class=argparse.RawTextHelpFormatter)

# Required input_file name argument.
parser.add_argument('--input_files', required=False, nargs='+', default=[],
                    help='Strings representing analysis input files.')

args = parser.parse_args()
dfs = [None] * len(args.input_files)
# Loads all the input files.
for index, input_file in enumerate(args.input_files):
    df = pd.read_csv(input_file)
    df['confs'] = ['c' + str(x) for x in range(len(df))]
    dfs[index] = df


for index, df in enumerate(dfs):
    plt.plot(df['confs'], df['result'], marker='o', label=args.input_files[index])
plt.legend()
plt.xlabel("confs")
plt.ylabel("result")
plt.show()
