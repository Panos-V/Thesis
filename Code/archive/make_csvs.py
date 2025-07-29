import pandas as pd
import os

df = pd.read_csv('full_data.csv')
train = pd.read_csv('train_metadata.csv')
test = pd.read_csv('test_metadata.csv')

merge_folder = 'merge'
train_folder = 'train'

def get_image_names(folder):
    files = os.listdir(folder)
    # Remove extension for matching with image_name column
    names = [os.path.splitext(f)[0] for f in files if not f.startswith('.')]
    return set(names)

def get_augmented_images(folder):
    files = os.listdir(folder)
    # Only files matching the augmented pattern
    aug_files = [f for f in files if '_aug' in f and not f.startswith('.')]
    return aug_files

def get_original_name(aug_name):
    # Remove extension, then split at '_aug_'
    base = os.path.splitext(aug_name)[0]
    return base.split('_aug')[0]

def make_augmented_df(folder):
    aug_files = get_augmented_images(folder)
    rows = []
    for aug_file in aug_files:
        orig_name = get_original_name(aug_file)
        # Find the row in df with this original image name
        match = df[df['image_name'] == orig_name]
        if not match.empty:
            row = match.iloc[0].copy()
            row['image_name'] = os.path.splitext(aug_file)[0]  # keep augmented name without extension
            rows.append(row)
    return pd.DataFrame(rows)

aug_train_folder = 'augmented_train'
aug_test_folder = 'augmented_test'

aug_train_df = make_augmented_df(aug_train_folder).reset_index(drop=True)
aug_test_df = make_augmented_df(aug_test_folder).reset_index(drop=True)

new_train = pd.concat([train, aug_train_df], ignore_index=True)
new_test = pd.concat([test, aug_test_df], ignore_index=True)

new_train['mode'] = 'train'
new_test['mode'] = 'test'

final_df = pd.concat([new_train, new_test], ignore_index=True)
final_df.to_csv('metadata_final.csv', index=False)

""" print(new_test)
print(test)
print(new_train)
print(train) """

""" merge_images = get_image_names(merge_folder)
train_images = get_image_names(train_folder)

# Filter the dataframe for images present in each folder
merge_df = df[df['image_name'].isin(merge_images)]
train_df = df[df['image_name'].isin(train_images)]

# Save to new CSV files
merge_df.to_csv('merge_in_full_data.csv', index=False)
train_df.to_csv('train_in_full_data.csv', index=False)

print(f"merge_in_full_data.csv: {len(merge_df)} rows")
print(f"train_in_full_data.csv: {len(train_df)} rows") """