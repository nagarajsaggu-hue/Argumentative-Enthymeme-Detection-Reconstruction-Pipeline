import pandas as pd

# Paths
STEP4A_PATH = "/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project/data/processed/mtg_step4a_adu_candidates.csv"
STEP4B_PATH = "/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project/data/processed/mtg_step4b_final_enthymemes.csv"

def patch():
    print("Loading files...")
    df_4a = pd.read_csv(STEP4A_PATH)
    df_4b = pd.read_csv(STEP4B_PATH)

    # We need to map the 'removed_idx' back to the label it had in 4a
    # Assuming df_4a has a 'labels' column (list of ADU types per sentence)
    label_map = {}
    
    print("Mapping labels...")
    for _, row in df_4a.iterrows():
        # Convert string representation of list "[0, 1, 2]" to actual list
        labels = eval(row['adu_labels']) if isinstance(row['adu_labels'], str) else row['adu_labels']
        label_map[row['File']] = labels

    def get_label(row):
        file_id = row['File']
        idx = int(row['removed_idx'])
        if file_id in label_map:
            return label_map[file_id][idx]
        return 1 # Default to Premise if not found

    df_4b['adu_label'] = df_4b.apply(get_label, axis=1)
    
    df_4b.to_csv(STEP4B_PATH, index=False)
    print(f"✅ Patched {len(df_4b)} rows with ADU labels!")

if __name__ == "__main__":
    patch()
