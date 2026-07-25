"""
Bilingual Text Summarizer using Llama-3.2-3B-Instruct
Supports English and Hindi text summarization from files and user input
"""

import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class BilingualSummarizer:
    """
    A summarizer class that handles English and Hindi text summarization
    using Llama-3.2-3B-Instruct model with long context support.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 4-bit quantization config for reduced VRAM usage
        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model with token from .env and local cache only
        logger.info(f"Loading model: {model_name}..")
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=True,
        )
        
        if self.device == "cpu":
            self.model.to(self.device)
        
        logger.info("Model loaded successfully")
        print("Model loaded successfully!")
    
    def detect_language(self, text: str) -> str:
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = len(text)
        return 'hi' if total_chars > 0 and devanagari_count / total_chars > 0.3 else 'en'
    
    def chunk_text(self, text: str, max_tokens: int = 3000) -> list:
        tokens = self.tokenizer.encode(text)
        return [self.tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True) 
                for i in range(0, len(tokens), max_tokens)]
    
    def summarize(self, text: str, language: Optional[str] = None, summary_length: str = "medium", sentiment_data: Optional[dict]=None) -> dict:
        if language is None: 
            language = self.detect_language(text)
        
        # 1. Define systems with optional sentiment context injection
        systems = {
            'en': "You are a helpful assistant. Provide the response in exactly two parts: 'Summary:' and 'Sentiment:'",
            'hi': "आप एक सहायक हैं। कृपया निम्नलिखित पाठ का सारांश 'Summary:' और 'Sentiment:' के रूप में दें।"
        }

        system_context = systems[language]
        if sentiment_data:
            emotion = sentiment_data.get("dominant_emotion","neutral")
            logger.info(f"Dominent Emotion: {emotion}")
            intensity = sentiment_data.get("intensity_description", "moderate")
            logger.info(f"Intensity description: {intensity}")
            system_context += f" Tone Context: The input text conveys a {intensity} {emotion} tone."
            logger.info(f"System Context: {system_context}")
        
        prompts = {
            'en': {
                'short': "Summarize in 2-3 sentences, then provide a sentiment analysis:\n\n{text}", 
                'medium': "Summarize in 4-6 sentences, then provide a sentiment analysis:\n\n{text}", 
                'long': "Summarize in 8-10 sentences, then provide a sentiment analysis:\n\n{text}"
            },
            'hi': {
                'short': "निम्नलिखित पाठ का 2-3 वाक्यों में सारांश दें और फिर भावना विश्लेषण (sentiment analysis) करें:\n\n{text}", 
                'medium': "निम्नलिखित पाठ का 4-6 वाक्यों में सारांश दें और फिर भावना विश्लेषण (sentiment analysis) करें:\n\n{text}", 
                'long': "निम्नलिखित पाठ का 8-10 वाक्यों में विस्तृत सारांश दें और फिर भावना विश्लेषण (sentiment analysis) करें:\n\n{text}"
            }
        }
        
        # Check if text is too long and use hierarchical summarization
        text_tokens = len(self.tokenizer.encode(text))
        if text_tokens > 3500:  # Use hierarchical summarization for long texts
            print(f"Text is long ({text_tokens} tokens). Using hierarchical summarization...")
            return self._hierarchical_summarize(text, language, summary_length, prompts, sentiment_data=sentiment_data)
        
        # 2. Use the defined 'systems' dictionary
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": prompts[language][summary_length].format(text=text)}
        ]
        # 3. Apply chat template and get the formatted text
        formatted_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # 4. Tokenize properly and move to device (with truncation disabled to avoid silent data loss)
        inputs = self.tokenizer(formatted_text, return_tensors="pt", truncation=False).to(self.device)
        input_length = inputs['input_ids'].shape[-1]
        
        # 5. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 6. Extract only the generated tokens
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return {'summary': generated_text.strip(), 'language': language}
    
    def _hierarchical_summarize(self, text, language, summary_length, prompts, sentiment_data=None):
        chunks = self.chunk_text(text, max_tokens=2000)
        chunk_summaries = []
        prompt_key = 'short' if summary_length == 'short' else 'medium'

        system_instr = "आप एक सहायक हैं।" if language == 'hi' else "You are a helpful assistant."
        if sentiment_data:
            emotion = sentiment_data.get("dominant_emotion", "neutral")
            system_instr += f" Tone: {emotion}."
        
        for chunk in chunks:
            prompt = f"System: {system_instr}\n\n{prompts[language][prompt_key].format(text=chunk)}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs['input_ids'].shape[-1]
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=0.7, 
                    top_p=0.9, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            chunk_summaries.append(self.tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip())
            
        combined = "\n".join(chunk_summaries)
        prompt = prompts[language][summary_length].format(text=combined)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[-1]
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return {
            'summary': self.tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip(), 
            'language': language, 
            'text_length': len(self.tokenizer.encode(text)), 
            'summary_length': summary_length, 
            'model': self.model_name, 
            'chunks_processed': len(chunks)
        }

    def load_text_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: 
            return f.read()
    
    def summarize_file(self, file_path: str, language: Optional[str] = None, summary_length: str = "medium", sentiment_data: Optional[dict] = None) -> dict:
        # Pass the sentiment_data argument here!
        result = self.summarize(self.load_text_file(file_path), language, summary_length, sentiment_data=sentiment_data)
        result['source_file'] = file_path
        return result

    def save_summary(self, summary_result: dict, output_file: str) -> None:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"SUMMARY:\n\n{summary_result['summary']}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--language", choices=['en', 'hi', 'auto'], default='auto')
    parser.add_argument("--length", choices=['short', 'medium', 'long'], default='medium')
    parser.add_argument("--output", type=str)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    
    args = parser.parse_args()
    
    if not args.file and not args.text: 
        parser.error("Provide --file or --text")
    
    summarizer = BilingualSummarizer(model_name=args.model)
    lang = None if args.language == 'auto' else args.language
    
    if args.file:
        result = summarizer.summarize_file(args.file, lang, args.length)
    else:
        result = summarizer.summarize(args.text, lang, args.length)
        
    print(f"\n{result['summary']}")
    if args.output: 
        summarizer.save_summary(result, args.output)

if __name__ == "__main__":
    main()