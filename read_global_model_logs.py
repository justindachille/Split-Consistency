import pandas as pd
from ast import literal_eval
import argparse
import numpy as np

def calc_stats(group):
    stats_dict = {
        'mean_global_accuracy': group['Best Global Model Test'].mean() * 100,
        'std_global_accuracy': group['Best Global Model Test'].std() * 100,
        'mean_global_top5_accuracy': group['Best Global Model Test Top-5'].mean() * 100,
        'std_global_top5_accuracy': group['Best Global Model Test Top-5'].std() * 100,
        'runs': group.shape[0]
    }
    
    return pd.Series(stats_dict)

# Set display options to show full DataFrame without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Define an argument parser and add arguments
parser = argparse.ArgumentParser(description='Process log files.')
parser.add_argument('--filename', type=str, help='CSV filename')
parser.add_argument('--alg', type=str, help='Algorithm name')
parser.add_argument('--dataset_filter', type=str, help='Dataset to filter by')
parser.add_argument('--partition', type=str, help='Partition to filter by', required=True)
# parser.add_argument('--alpha', type=float, help='Alpha value to filter by', required=True)
args = parser.parse_args()

# Load the CSV file into a DataFrame
df = pd.read_csv(args.filename)

def parse_hyperparameters(hyperparams):
    return literal_eval(hyperparams)

# Parse the 'Hyperparameters' column to a dictionary
df['Params'] = df['Hyperparameters'].apply(parse_hyperparameters)

# Create a column for 'alpha' by extracting the value from the 'Params' column
# df['alpha'] = df['Params'].apply(lambda x: x.get('alpha'))

# Create a column for 'split_layer' by extracting the value from the 'Params' column
df['split_layer'] = df['Params'].apply(lambda x: x.get('split_layer'))

# Create a column for 'partition' by extracting the value from the 'Params' column
df['partition'] = df['Params'].apply(lambda x: x.get('partition'))

# Apply filters based on dataset, partition, alpha, and algorithm
df = df[(df['Params'].apply(lambda x: x['dataset'] == args.dataset_filter)) &
        (df['partition'] == args.partition) &
        # (df['alpha'] == args.alpha) &
        (df['Params'].apply(lambda x: x['alg'] == args.alg)) &
        (df['Client ID'] == 0)]

print("Filtered DataFrame based on specified criteria:")
print(df.to_string(index=False))

groupby_columns = ['partition', 'split_layer']
grouped = df.groupby(groupby_columns)

# Calculate the stats for each group
stats = grouped.apply(calc_stats).reset_index()

def getTitle(alg, split_layer):
    algorithm = ""
    if args.alg == "local_training":
        algorithm = "Local Training"
    elif args.alg == 'sflv1':
        algorithm = f"SFL-V1 ($L_c={split_layer}$)"
    elif args.alg == 'sflv2':
        algorithm = f"SFL-V2 ($L_c={split_layer}$)"
    elif args.alg == 'fedavg':
        algorithm = "Fedavg"
    elif args.alg == 'fedprox':
        algorithm = "Fedprox"
    elif args.alg == 'moon':
        algorithm = 'MOON'
    else:
        assert(ValueError("Bad alg input"))
    return algorithm

def print_latex_table(stats_df, partition):
    print(f"% Results for partition={partition}")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Training results for partition={partition}:}}")
    print("\\begin{adjustbox}{max width=\\textwidth}")
    print("\\begin{tabular}{|l|c|c|c|} \\hline")
    headers = "Algorithm & Global Test Acc. & Global Top5 Test Acc. & Runs \\\\ \\hline"
    print(headers)
    for _, row in stats_df.iterrows():
        split_layer = int(row['split_layer'])
        algorithm = getTitle(args.alg, split_layer)
        global_test_acc = f"{row['mean_global_accuracy']:.2f} ± {row['std_global_accuracy']:.2f}"
        global_top5_test_acc = f"{row['mean_global_top5_accuracy']:.2f} ± {row['std_global_top5_accuracy']:.2f}"
        runs = int(row['runs'])
        print(f"{algorithm} & {global_test_acc} & {global_top5_test_acc} & {runs} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{adjustbox}")
    print("\\end{table}")

# Print LaTeX tables for varying partitions
for partition in stats['partition'].unique():
    partition_stats = stats[stats['partition'] == partition]
    print_latex_table(partition_stats, partition)