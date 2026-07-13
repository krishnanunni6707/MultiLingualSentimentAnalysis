import pandas as pd

def clean(raw_dataset,cleaned_dataset):
    try:
        df = pd.read_csv(raw_dataset)
        df_cleaned = df.fillna(0)
        df_cleaned.to_csv(cleaned_dataset,index=False)
        print("Cleaned file Successfully!!")
    except FileNotFoundError:
        print('Error: File not found')
    except Exception as e:
        print(f'An error  occured: {e}')

clean('eng-emo-dataset.csv','cleaned-dataset.csv')