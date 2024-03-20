import pandas as pd
from ast import literal_eval
import argparse
import numpy as np

def calc_stats(group):
    stats_dict = {
        'mean_global_accuracy': group['Best Global Accuracy'].mean() * 100,
        'std_global_accuracy': group['Best Global Accuracy'].std() * 100,
        'mean_global_top5_accuracy': group['Best Global Accuracy Top-5'].mean() * 100,
        'std_global_top5_accuracy': group['Best Global Accuracy Top-5'].std() * 100,
        'mean_best_global_model_train': group['Best Global Model Train'].mean() * 100,
        'std_best_global_model_train': group['Best Global Model Train'].std() * 100,
        'mean_best_global_model_test': group['Best Global Model Test'].mean() * 100,
        'std_best_global_model_test': group['Best Global Model Test'].std() * 100,
        'runs': group.shape[0]
    }
    
    for client_number in group['Client ID'].unique():
        client_group = group[group['Client ID'] == client_number]
        client_name = f"Client A" if client_number == 0 else f"Client B"
        stats_dict.update({
            f'mean_local_accuracy_{client_name}': client_group['Best Local Accuracy'].mean() * 100,
            f'std_local_accuracy_{client_name}': client_group['Best Local Accuracy'].std() * 100,
            f'mean_local_top5_accuracy_{client_name}': client_group['Best Local Accuracy Top-5'].mean() * 100,
            f'std_local_top5_accuracy_{client_name}': client_group['Best Local Accuracy Top-5'].std() * 100,
            f'mean_global_accuracy_{client_name}': client_group['Best Global Accuracy'].mean() * 100,
            f'std_global_accuracy_{client_name}': client_group['Best Global Accuracy'].std() * 100,
        })
        
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
parser.add_argument('--alpha', type=float, help='Alpha value to filter by', required=True)
parser.add_argument('--table_format', action='store_true', help='Print results in table format')
args = parser.parse_args()

# Load the CSV file into a DataFrame
df = pd.read_csv(args.filename)

def parse_hyperparameters(hyperparams):
    return literal_eval(hyperparams)

# Parse the 'Hyperparameters' column to a dictionary
df['Params'] = df['Hyperparameters'].apply(parse_hyperparameters)

# Create a column for 'alpha' by extracting the value from the 'Params' column
df['alpha'] = df['Params'].apply(lambda x: x.get('alpha'))

# Apply filters based on dataset, alpha, and algorithm
df = df[(df['Params'].apply(lambda x: x['dataset'] == args.dataset_filter)) &
        (df['alpha'] == args.alpha) &
        (df['Params'].apply(lambda x: x['alg'] == args.alg))]

print("Filtered DataFrame based on specified criteria:")
print(df.to_string(index=False))

groupby_columns = ['alpha']
if 'split_layer' in df.columns:
    groupby_columns.append('split_layer')
grouped = df.groupby(groupby_columns)

# Calculate the stats for each group
stats = grouped.apply(calc_stats).reset_index()

if args.table_format:
    print("\\textbf{Algorithms} & Global model accuracy & Client A's local / global FT accuracy & Client B's local / global FT accuracy\\\\ \\hline")
    for _, row in stats.iterrows():
        algorithm = f"{args.alg} {row['alpha']}"
        global_model_acc = f"{row['mean_best_global_model_test']:.2f} ± {row['std_best_global_model_test']:.2f}"
        client_a_local_global = f"{row['mean_local_accuracy_Client A']:.2f} ± {row['std_local_accuracy_Client A']:.2f} / {row['mean_global_accuracy_Client A']:.2f} ± {row['std_global_accuracy_Client A']:.2f}"
        client_b_local_global = f"{row['mean_local_accuracy_Client B']:.2f} ± {row['std_local_accuracy_Client B']:.2f} / {row['mean_global_accuracy_Client B']:.2f} ± {row['std_global_accuracy_Client B']:.2f}"
        print(f"{algorithm} & {global_model_acc} & {client_a_local_global} & {client_b_local_global} \\\\ \\hline")
else:
    def print_latex_table(stats_df, alpha_value):
        print(f"% Results for alpha={alpha_value}")
        print("\\begin{table}[h]")
        print("\\centering")
        print(f"\\caption{{Training results for alpha={alpha_value}:}}")
        print("\\begin{adjustbox}{max width=\\textwidth}")
        print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|} \\hline")
        headers = "Algorithm & Global Test Acc. & Global Top5 Test Acc. & Global Train & Global Test & Client A Local \\textbar{} Top5 \\textbar{} Global & Client B Local \\textbar{} Top5 \\textbar{} Global \\\\ \\hline"
        print(headers)
        for _, row in stats.iterrows():
            algorithm = f"{args.alg} {alpha_value}"
            global_test_acc = f"{row['mean_global_accuracy']:.2f} ± {row['std_global_accuracy']:.2f}%"
            global_top5_test_acc = f"{row['mean_global_top5_accuracy']:.2f} ± {row['std_global_top5_accuracy']:.2f}%"
            global_train = f"{row['mean_best_global_model_train']:.2f} ± {row['std_best_global_model_train']:.2f}%"
            global_test = f"{row['mean_best_global_model_test']:.2f} ± {row['std_best_global_model_test']:.2f}%"
            client_a_local_top5_global = f"{row['mean_local_accuracy_Client A']:.2f} ± {row['std_local_accuracy_Client A']:.2f}% \\textbar{{}} {row['mean_local_top5_accuracy_Client A']:.2f} ± {row['std_local_top5_accuracy_Client A']:.2f}% \\textbar{{}} {row['mean_global_accuracy_Client A']:.2f} ± {row['std_global_accuracy_Client A']:.2f}%"
            client_b_local_top5_global = f"{row['mean_local_accuracy_Client B']:.2f} ± {row['std_local_accuracy_Client B']:.2f}% \\textbar{{}} {row['mean_local_top5_accuracy_Client B']:.2f} ± {row['std_local_top5_accuracy_Client B']:.2f}% \\textbar{{}} {row['mean_global_accuracy_Client B']:.2f} ± {row['std_global_accuracy_Client B']:.2f}%"
            print(f"{algorithm} & {global_test_acc} & {global_top5_test_acc} & {global_train} & {global_test} & {client_a_local_top5_global} & {client_b_local_top5_global} \\\\ \\hline")
        print("\\end{tabular}")
        print("\\end{adjustbox}")
        print("\\end{table}")
    # Print LaTeX tables for varying alpha
    for alpha_value in stats['alpha'].unique():
        print_latex_table(stats, alpha_value)