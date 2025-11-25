import csv
import numpy as np

csv_path = "logs/eval_log_20240221_123456.csv"   # <-- replace with any log

# Read all rows
rows = []
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)      # first row is header
    for row in reader:
        rows.append(row)

rows = np.array(rows, dtype=float)   # convert to numpy array

print("Loaded CSV with shape:", rows.shape)
print("Columns:", header)
