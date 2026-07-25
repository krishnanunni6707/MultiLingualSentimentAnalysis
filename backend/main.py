import sys
from utils.logger import setup_logger
from utils.logger import get_logger
setup_logger(log_file="logs/app.log", force=True)
import argparse
import torch
from pathlib import Path
from config import APIConfig
from inference.pipeline import RealTimeInference

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Sentiment & Summarization CLI")
    parser.add_argument("--file", type=str, help="Path to text file")
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--language", choices=['hi', 'en'], help="Language code")
    parser.add_argument("--length", choices=['short', 'medium', 'long'], default='medium')
    args = parser.parse_args()

    if not args.file and not args.text:
        print("Error: Provide --file or --text")
        return

    # 1. Get input text first
    print("[~] Reading input...")
    text = ""
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text

    # 2. Load all models eagerly at startup (sentiment + translator + summarizer)
    print("[~] Loading all models...")
    inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)
    
    # 3. Run sentiment analysis (pipeline internally runs the summarizer too)
    print("[~] Running sentiment analysis...")
    sentiment_result = inference.predict(text, language=args.language)

    # 4. Output
    overall = sentiment_result.get("overall_sentiment", {})
    metadata = sentiment_result.get("metadata", {})
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Language: {metadata.get('language', args.language or 'auto')}")
    print(f"Dominant Emotion: {overall.get('dominant_emotion')}")
    print(f"Polarity: {overall.get('polarity_description')}")
    print(f"\n{'SUMMARY':-^50}")
    print(f"{sentiment_result.get('summary', 'N/A')}")
    print("="*50)

if __name__ == "__main__":
    logger.info("Starting the Readers Digest Engine....")
    logger.info("Pre loading the Custom Models")
    main()