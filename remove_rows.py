import csv
import os
import sys


def filter_csv_inplace(file_path):
    temp_file = 'temp.csv'

    with open(file_path, 'r') as file, open(temp_file, 'w', newline='') as temp:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(temp, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if int(row['Client ID']) == 0:
                writer.writerow(row)

    os.replace(temp_file, file_path)
    print(f"CSV file updated: {file_path}")

# Usage example
file_path = sys.argv[1]

filter_csv_inplace(file_path)