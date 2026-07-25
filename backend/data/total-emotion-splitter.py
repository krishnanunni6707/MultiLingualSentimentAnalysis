import pandas as pd

# Load the dataset from your CSV file
# Replace 'your_dataset.csv' with the actual path to your file
df = pd.read_csv('EmoInHindi/LREC_EmoInHindi_reduced.csv')

# Extract all unique emotions
# 1. Convert to string and split by comma
# 2. Use a set comprehension to collect unique values while removing whitespace
unique_emotions = {
    emotion.strip() 
    for row in df['emotions'].astype(str) 
    for emotion in row.split(',')
}

# Convert set to a sorted list for better readability
unique_emotions_list = sorted(list(unique_emotions))

print(f"Total Unique Emotions Found: {len(unique_emotions_list)}")
print(f"List of Unique Emotions: {unique_emotions_list}")