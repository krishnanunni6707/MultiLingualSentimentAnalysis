import sys
from utils.logger import setup_logger
from utils.logger import get_logger

setup_logger(log_file="logs/app.log", force=True)

import argparse
import torch
from pathlib import Path
from config import APIConfig
from inference.pipeline import RealTimeInference

import edge_tts
import os
import asyncio

logger = get_logger(__name__)

#==============> Voice Configurations <===========
OUTPUT_FILE = "temp_voice_output.mp3"

async def Speak(text, VOICE):
    com = edge_tts.Communicate(text, VOICE)
    await com.save(OUTPUT_FILE)
    os.system(f"mpv --no-video {OUTPUT_FILE}")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

def process_text(inference, text, default_language):
    
    lang_param = None if default_language == 'auto' else default_language
    try:
        sentiment_result = inference.predict(text, language=lang_param)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        print(f"Error during analysis: {e}")
        return

    overall = sentiment_result.get("overall_sentiment", {})
    metadata = sentiment_result.get("metadata", {})
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Language: {metadata.get('language', default_language or 'auto')}")
    print(f"Dominant Emotion: {overall.get('dominant_emotion')}")
    print(f"Polarity: {overall.get('polarity_description')}")
    print(f"\n{'SUMMARY':-^50}")
    print(f"{sentiment_result.get('summary', 'N/A')}")
    print("="*50)
    
    try:
        check = input("Need to Read!! (y/n)?").strip()
    except (KeyboardInterrupt, EOFError):
        return
        
    txt = sentiment_result.get('summary', 'N/A')
    lang = metadata.get('language', default_language or 'auto')
    if check == 'y' or check == 'Y':
        logger.info("[~] Speach Activated")
        if lang == 'hi':
            VOICE = "hi-IN-MadhurNeural"
            asyncio.run(Speak(txt, VOICE))
        elif lang == 'en':
            VOICE = "en-US-AndrewNeural"
            asyncio.run(Speak(txt, VOICE))
        else:
            logger.info("[~] Speach for this language is not defined!!")
            print("[~] Speach for this language is not defined!!")

def main():
    parser = argparse.ArgumentParser(description="Sentiment & Summarization CLI")
    parser.add_argument("--file", type=str, help="Path to text file")
    parser.add_argument("--text", type=str, help="Inline text to analyze")
    parser.add_argument("--language", type=str, help="Language code ('hi', 'en', 'auto')", default='auto')
    parser.add_argument("--length", choices=['short', 'medium', 'long'], default='medium')
    args = parser.parse_args()

    print("")
    inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)
    initial_text = ""
    if args.file:
        print(f"[~] Reading file: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                initial_text = f.read()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
    elif args.text:
        initial_text = args.text

    if initial_text:
        process_text(inference, initial_text, args.language)

    # 3. Enter interactive loop
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nEnter text to analyze (or file path, or 'exit' to quit): ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
            
            if os.path.exists(user_input):
                print(f"[~] Reading file: {user_input}")
                try:
                    with open(user_input, 'r', encoding='utf-8') as f:
                        user_input = f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            
            process_text(inference, user_input, args.language)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    logger.info("Starting the Readers Digest Engine....")
    logger.info("[~] Loading all models...")
    main()