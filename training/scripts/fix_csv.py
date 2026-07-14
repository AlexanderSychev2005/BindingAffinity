import os
import pandas as pd


def main():
    data_dir = "refined-set"
    csv_file = "pdbbind_refined_dataset.csv"

    # Get all valid directory names
    valid_dirs = set(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    df = pd.read_csv(csv_file, dtype={"pdb_id": str})

    csv_ids = set(df["pdb_id"].astype(str))

    # Find corrupted IDs
    corrupted_ids = []
    for pid in df["pdb_id"]:
        if str(pid) not in valid_dirs:
            corrupted_ids.append(str(pid))

    # Find missing valid IDs (the true forms of the corrupted ones)
    missing_valid_ids = valid_dirs - csv_ids

    print(f"Found {len(corrupted_ids)} corrupted IDs in CSV.")
    print(f"Found {len(missing_valid_ids)} valid directories not in CSV.")

    # Mapping
    correction_map = {}

    for cid in corrupted_ids:
        # Try to match
        matched = False

        # 1. Scientific notation fix
        if "E+" in cid:
            try:
                parts = cid.split("E+")
                guess = str(int(float(parts[0]))) + "e" + parts[1]
                if guess in missing_valid_ids:
                    correction_map[cid] = guess
                    missing_valid_ids.remove(guess)
                    matched = True
            except:
                pass

        # 2. Date fix (e.g. 4-Oct -> 4oct)
        if not matched and "-" in cid:
            guess = cid.lower().replace("-", "")
            if guess in missing_valid_ids:
                correction_map[cid] = guess
                missing_valid_ids.remove(guess)
                matched = True
            else:
                # Sometimes it might be Oct-04 -> 4oct
                parts = cid.lower().split("-")
                if len(parts) == 2:
                    guess2 = parts[1].lstrip("0") + parts[0]
                    if guess2 in missing_valid_ids:
                        correction_map[cid] = guess2
                        missing_valid_ids.remove(guess2)
                        matched = True
                    else:
                        guess3 = parts[0].lstrip("0") + parts[1]
                        if guess3 in missing_valid_ids:
                            correction_map[cid] = guess3
                            missing_valid_ids.remove(guess3)
                            matched = True

        # 3. Last resort fallback (try all remaining and pick closest)
        if not matched:
            print(f"Could not automatically match {cid}")

    print("Correction map:", correction_map)

    # Apply corrections
    df["pdb_id"] = df["pdb_id"].astype(str).replace(correction_map)

    # Save fixed CSV
    fixed_csv = "pdbbind_refined_dataset.csv"
    df.to_csv(fixed_csv, index=False)
    print(f"Fixed CSV saved to {fixed_csv}")


if __name__ == "__main__":
    main()
