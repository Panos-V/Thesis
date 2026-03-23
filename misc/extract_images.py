import pandas as pd
import requests

def extract_images_from_csv(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    
    for index, row in df.iterrows():
        url = row['url']  # Assuming the column name is 'url'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        with open(f'{output_dir}/{index}.jpg','wb') as f:
            response = requests.get(url, headers=headers, stream=True)
            if not response.ok:
                print(response)
            
            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)

if __name__ == "__main__":
    csv_file = 'raw_data/Fitzpatrick17k/fitzpatrick17k.csv'  # Path to your CSV file
    output_dir = 'raw_data/Fitzpatrick17k'  # Directory to save downloaded images
    extract_images_from_csv(csv_file, output_dir)