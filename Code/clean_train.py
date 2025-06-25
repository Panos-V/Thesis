import os
import pandas as pd


directory = 'Code/archive/train/'
deleted = 0
for fname in os.listdir(directory):
    fpath = os.path.join(directory, fname)
    if os.path.isfile(fpath) and '_AUGMENTATION_' in fname:
        try:
            os.remove(fpath)
            print(f"Deleted: {fpath}")
            deleted += 1
        except Exception as e:
            print(f"Error deleting {fpath}: {e}")
print(f"\nFinished! Deleted {deleted} files containing 'augmented' in their name.")

df = pd.read_csv('Code/archive/original.csv')
df_clean = df[~df['image_name'].str.contains('AUGMENTATION', case=True, regex=False)]
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv("Code/archive/original.csv",index=False)