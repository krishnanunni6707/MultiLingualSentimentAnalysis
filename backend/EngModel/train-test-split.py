import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = 'cleaned-dataset.csv' 
output_dir = './' 

def process_and_split(file_path):
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    df = pd.read_csv(file_path)
    
    df = df.fillna(0)
    
    X = df[['text']]
    y = df.drop(columns=['text'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    missing_in_train = [col for col in y_train.columns if y_train[col].sum() == 0]
    
    if missing_in_train:
        print(f"Warning: The following emotions are missing in the training set: {missing_in_train}")
    else:
        print("Success: All emotion categories are represented in the training set.")
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Split complete.")
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")

if __name__ == "__main__":
    process_and_split(input_file)