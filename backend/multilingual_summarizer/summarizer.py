"""Gemini-backed bilingual text summarizer."""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class BilingualSummarizer:
    """Summarize English and Hindi text using Google's Gemini API."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash"
        )
        self.fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"
        )
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Gemini API key is missing. Set GEMINI_API_KEY in backend/.env."
            )

        self.client = genai.Client(api_key=self.api_key)
        logger.info("Gemini summarizer configured with model: %s", self.model_name)

    def detect_language(self, text: str) -> str:
        devanagari_count = sum(1 for char in text if "\u0900" <= char <= "\u097F")
        return (
            "hi"
            if text and devanagari_count / len(text) > 0.3
            else "en"
        )

    def summarize(
        self,
        text: str,
        language: Optional[str] = None,
        summary_length: str = "medium",
        sentiment_data: Optional[dict] = None,
    ) -> dict:
        if not text.strip():
            raise ValueError("Cannot summarize empty text.")

        language = language or self.detect_language(text)
        if language not in {"en", "hi"}:
            raise ValueError(f"Unsupported summarization language: {language}")
        if summary_length not in {"short", "medium", "long"}:
            raise ValueError(f"Unsupported summary length: {summary_length}")

        sentence_count = {"short": "2-3", "medium": "4-6", "long": "8-10"}[
            summary_length
        ]
        output_language = "Hindi" if language == "hi" else "English"
        sentiment_context = ""
        if sentiment_data:
            emotion = sentiment_data.get("dominant_emotion", "neutral")
            intensity = sentiment_data.get("intensity_description", "moderate")
            sentiment_context = (
                f"\nThe detected tone is {intensity} {emotion}. "
                "Reflect this tone accurately without inventing facts."
            )

        # Keep requests small enough for a predictable API response on long
        # documents while retaining the beginning and conclusion.
        max_input_chars = 12000
        bounded_text = text if len(text) <= max_input_chars else (
            text[:9000] + "\n\n[ middle of document omitted ]\n\n" + text[-3000:]
        )
        prompt = (
            f"Summarize the following text in {sentence_count} sentences in "
            f"{output_language}. Include the main idea and important details. "
            "Return only the summary, without headings or preamble."
            f"{sentiment_context}\n\nText:\n{bounded_text}"
        )
        generation_config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens={"short": 256, "medium": 512, "long": 1024}[
                summary_length
            ],
        )
        response = None
        models = [self.model_name]
        if self.fallback_model != self.model_name:
            models.append(self.fallback_model)
        last_error = None
        for model in models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=generation_config,
                    )
                    self.model_name = model
                    break
                except Exception as error:
                    last_error = error
                    message = str(error).lower()
                    transient = any(
                        marker in message
                        for marker in ("503", "unavailable", "timed out", "timeout")
                    )
                    if not transient or attempt == 1:
                        break
                    time.sleep(1.5 * (attempt + 1))
            if response is not None:
                break
        if response is None:
            raise RuntimeError(
                f"Gemini summarization failed after retries: {last_error}"
            ) from last_error
        summary = (response.text or "").strip()
        if not summary:
            raise RuntimeError("Gemini returned an empty summary.")

        return {
            "summary": summary,
            "language": language,
            "summary_length": summary_length,
            "model": self.model_name,
        }

    def load_text_file(self, file_path: str) -> str:
        return Path(file_path).read_text(encoding="utf-8")

    def summarize_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        summary_length: str = "medium",
        sentiment_data: Optional[dict] = None,
    ) -> dict:
        result = self.summarize(
            self.load_text_file(file_path),
            language,
            summary_length,
            sentiment_data=sentiment_data,
        )
        result["source_file"] = file_path
        return result

    def save_summary(self, summary_result: dict, output_file: str) -> None:
        Path(output_file).write_text(
            f"SUMMARY:\n\n{summary_result['summary']}\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--language", choices=["en", "hi", "auto"], default="auto")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--output", type=str)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if not args.file and not args.text:
        parser.error("Provide --file or --text")

    summarizer = BilingualSummarizer(model_name=args.model)
    language = None if args.language == "auto" else args.language
    result = (
        summarizer.summarize_file(args.file, language, args.length)
        if args.file
        else summarizer.summarize(args.text, language, args.length)
    )
    print(f"\n{result['summary']}")
    if args.output:
        summarizer.save_summary(result, args.output)


if __name__ == "__main__":
    main()
