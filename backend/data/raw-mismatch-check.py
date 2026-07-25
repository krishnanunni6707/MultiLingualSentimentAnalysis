import pandas as pd

def check_emotion_intensity_mismatch(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    mismatches = []
    
    for idx, row in df.iterrows():
        # Safely handle missing values by defaulting to empty strings
        emo_str = str(row.get("emotions", "")) if pd.notna(row.get("emotions")) else ""
        intensity_str = str(row.get("emoIntensity", "")) if pd.notna(row.get("emoIntensity")) else ""
        
        # Split by commas and filter out empty strings
        labels = [l.strip() for l in emo_str.split(",") if l.strip()]
        intensities = [i.strip() for i in intensity_str.split(",") if i.strip()]
        
        # If there's exactly 1 intensity for multiple emotions, your dataset class duplicates it.
        # We exclude that case from being an error, or you can remove the second condition if you want strict equality.
        if len(labels) != len(intensities) and not (len(intensities) == 1 and len(labels) > 1):
            mismatches.append({
                "row_index": idx,
                "dialogueId": row.get("dialogueId", "N/A"),
                "utterance": row.get("utterance", ""),
                "emotions": emo_str,
                "emoIntensity": intensity_str,
                "emotion_count": len(labels),
                "intensity_count": len(intensities)
            })
            
    # Print results
    if mismatches:
        print(f"Found {len(mismatches)} rows with count mismatches:\n")
        for m in mismatches:
            print(f"Row {m['row_index']} (ID: {m['dialogueId']}):")
            print(f"  Utterance: {m['utterance']}")
            print(f"  Emotions ({m['emotion_count']}): {m['emotions']}")
            print(f"  Intensities ({m['intensity_count']}): {m['emoIntensity']}")
            print("-" * 50)
    else:
        print("All rows have matching emotion and intensity counts!")

# Run the check
csv_file_path = "EmoInHindi/emo-hindi-dataset.csv"
check_emotion_intensity_mismatch(csv_file_path)