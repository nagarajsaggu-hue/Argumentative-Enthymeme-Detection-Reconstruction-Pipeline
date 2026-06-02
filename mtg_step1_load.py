import pandas as pd
import numpy as np

PATH = "/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project/data/processed/mtg_step4b_final_enthymemes.csv"

def repair():
    df = pd.read_csv(PATH)
    print(f"Repairing {len(df)} rows...")

    def assign_adu(row):
        idx = row['removed_idx']
        total = row['num_sents']
        
        # Stahl et al. / Chen et al. Logic:
        # 1. Major Claims are usually at the very beginning or end.
        # 2. Claims are usually concluding or introductory statements.
        # 3. Premises are the 'body' of the argument.
        
        if idx == 0:
            return 3 # MajorClaim (~5-6% of data)
        elif idx == total - 1:
            return 2 # Claim (~30% of data)
        else:
            return 1 # Premise (~63% of data)

    df['adu_label'] = df.apply(assign_adu, axis=1)
    df.to_csv(PATH, index=False)
    print(" Done! 'adu_label' column added based on structural role.")

if __name__ == "__main__":
    repair()
