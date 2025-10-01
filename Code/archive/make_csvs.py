import pandas as pd
import os

train = pd.read_csv('train_metadata.csv')
test = pd.read_csv('test_metadata.csv')

def get_augmented_images(folder):
    files = os.listdir(folder)
    # Match any file with '_aug' or '_V2' in the name
    aug_files = [f for f in files if (('_aug' in f or '_V2' in f) and not f.startswith('.'))]
    return aug_files

def get_original_name(aug_name):
    # Remove extension, then split at '_aug' or '_V2'
    base = os.path.splitext(aug_name)[0]
    if '_aug' in base:
        return base.split('_aug')[0]
    elif '_V2' in base:
        return base.split('_V2')[0]
    else:
        return base

def make_augmented_df(folder, meta_df):
    aug_files = get_augmented_images(folder)
    rows = []
    for aug_file in aug_files:
        orig_name = get_original_name(aug_file)
        match = meta_df[meta_df['image_name'] == orig_name]
        if not match.empty:
            row = match.iloc[0].copy()
            row['image_name'] = os.path.splitext(aug_file)[0]
            rows.append(row)
    return pd.DataFrame(rows)

aug_train_folder = 'train'
aug_test_folder = 'test'

aug_train_df = make_augmented_df(aug_train_folder, train).reset_index(drop=True)
aug_test_df = make_augmented_df(aug_test_folder, test).reset_index(drop=True)

new_train = pd.concat([train, aug_train_df], ignore_index=True)
new_test = pd.concat([test, aug_test_df], ignore_index=True)

new_train['mode'] = 'train'
new_test['mode'] = 'test'

new_train.to_csv('train_metadata_with_aug.csv', index=False)
new_test.to_csv('test_metadata_with_aug.csv', index=False)

print(f"Train CSV with augmentations: {new_train.shape}")
print(f"Test CSV with augmentations: {new_test.shape}")