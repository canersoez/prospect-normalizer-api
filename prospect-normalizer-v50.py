import argparse
import pandas as pd
from rapidfuzz import fuzz, process

# (Your full normalization logic can go here, keeping a placeholder for now)

def normalize_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    # Apply your normalization logic here...
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize prospect list.")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()

    normalize_csv(args.input, args.output)
