import sys
from utils.logger import setup_logger
setup_logger(log_file="logs/app.log", force=True)
import argparse
import torch
from pathlib import Path
from config import APIConfig
from inference.pipeline import RealTimeInference
from multilingual_summarizer.summarizer import BilingualSummarizer

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

    # 2. STAGE 1: Load and run Sentiment Model
    print("[~] Loading sentiment model...")
    inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)
    
    print("[~] Running sentiment analysis...")
    sentiment_result = inference.predict(text, language=args.language)
    
    # 3. CLEANUP: Free GPU memory before loading summarizer
    print("[~] Cleaning up sentiment model from GPU...")
    del inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. STAGE 2: Load and run Summarization Model
    print("[~] Loading summarizer model...")
    summarizer = BilingualSummarizer()
    
    print("[~] Generating summary...")
    summary_result = summarizer.summarize(
        text=text,
        language=args.language,
        summary_length=args.length
    )

    # 5. Output
    overall = sentiment_result.get("overall_sentiment", {})
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Language: {sentiment_result.get('detected_language', args.language or 'auto')}")
    print(f"Dominant Emotion: {overall.get('dominant_emotion')}")
    print(f"Polarity: {overall.get('polarity_description')}")
    print(f"\n{'SUMMARY':-^50}")
    print(f"{summary_result['summary']}")
    print("="*50)

if __name__ == "__main__":
    main()