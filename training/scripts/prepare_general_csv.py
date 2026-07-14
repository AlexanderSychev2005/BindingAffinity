import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse PDBbind general index to CSV")
    parser.add_argument("--index-file", type=str, default="PDBbind_v2020_plain_text_index/index/INDEX_general_PL_data.2020", help="Path to index file")
    parser.add_argument("--out-csv", type=str, default="pdbbind_general_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.index_file):
        print(f"Error: Index file not found at {args.index_file}")
        return

    data = []
    with open(args.index_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                pdb_id = parts[0]
                try:
                    pkd = float(parts[3])
                    data.append({"pdb_id": pdb_id, "pKd": pkd})
                except ValueError:
                    print(f"Skipping {pdb_id} due to parsing error on {parts[3]}")

    df = pd.DataFrame(data)
    df.to_csv(args.out_csv, index=False)
    print(f"Successfully wrote {len(df)} entries to {args.out_csv}")

if __name__ == "__main__":
    main()
