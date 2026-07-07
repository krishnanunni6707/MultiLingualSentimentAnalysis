"""
CLI entrypoint for Hindi Sentiment Analysis
Provides interactive prompt and single-shot command line arguments.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Ensure backend directory is in the path
backend_path = Path(__file__).resolve().parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from config import LOG_DIR
from utils.logger import setup_logger

# Configure logging to go to file only to avoid polluting console output
setup_logger(level=logging.WARNING, log_file=str(LOG_DIR / "cli.log"), force=True)

from config import APIConfig
from inference.pipeline import RealTimeInference
logger = logging.getLogger(__name__)

# ANSI Escape Sequences for terminal coloring
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def color_text(text: str, color_code: str, enabled: bool = True) -> str:
    """Wrap text in ANSI color codes if enabled."""
    if enabled and sys.stdout.isatty():
        return f"{color_code}{text}{Colors.END}"
    return text


def print_banner(color_enabled: bool):
    """Print standard application welcome banner."""
    banner = f"""
========================================================================
   🌟 {color_text("Multi-Lingual Sentiment Analysis System (Hindi/English)", Colors.BOLD + Colors.CYAN, color_enabled)} 🌟
========================================================================
    """
    print(banner)


def format_sentiment_result(result: dict, text_to_analyze: str = None, detailed: bool = False, color_enabled: bool = True):
    """Print prediction results in a visually appealing, structured way."""
    if "error" in result:
        print(color_text(f"Error: {result['error']}", Colors.RED, color_enabled))
        return

    overall = result.get("overall_sentiment", {})
    metadata = result.get("metadata", {})
    segments = result.get("segments", [])

    print("-" * 72)
    print(color_text("ANALYSIS RESULTS SUMMARY", Colors.BOLD + Colors.CYAN, color_enabled))
    print("-" * 72)

    # Metadata details
    lang = metadata.get("language", "hi")
    lang_confidence = metadata.get("language_confidence", 1.0) * 100
    lang_str = "Hindi (hi)" if lang == "hi" else "English (en)" if lang == "en" else lang
    print(f"Detected Language:   {color_text(lang_str, Colors.BOLD, color_enabled)} (Confidence: {lang_confidence:.1f}%)")
    print(f"Original Length:     {metadata.get('original_length', 0)} characters")
    print(f"Number of Segments:  {metadata.get('num_segments', len(segments))}")

    # Polarity formatting
    polarity = overall.get("polarity_score", 0.0)
    polarity_desc = overall.get("polarity_description", "Neutral")
    if polarity > 0.05:
        pol_color = Colors.GREEN
    elif polarity < -0.05:
        pol_color = Colors.RED
    else:
        pol_color = Colors.YELLOW

    # Intensity formatting
    intensity = overall.get("intensity", 0.0)
    intensity_desc = overall.get("intensity_description", "Moderate")

    # Dominant Emotion
    dominant_emotion = overall.get("dominant_emotion", "unknown").upper()
    confidence = overall.get("confidence", 0.0) * 100

    print("\n" + color_text("OVERALL SENTIMENT:", Colors.BOLD, color_enabled))
    print(f"  Dominant Emotion:  {color_text(f'[ {dominant_emotion} ]', Colors.BOLD + Colors.MAGENTA, color_enabled)} (Confidence: {confidence:.1f}%)")
    print(f"  Polarity Score:    {color_text(f'{polarity:+.4f}', pol_color + Colors.BOLD, color_enabled)} ({color_text(polarity_desc, pol_color, color_enabled)})")
    print(f"  Intensity Score:   {color_text(f'{intensity:.4f}', Colors.BOLD, color_enabled)} ({intensity_desc})")

    # Cross-Check details
    cross_check = overall.get("cross_check_summary", {})
    if cross_check:
        num_segments = cross_check.get("num_segments", 1)
        num_agree = cross_check.get("num_agree", 1)
        agreement_rate = cross_check.get("agreement_rate", 1.0) * 100
        
        status_text = ""
        if agreement_rate >= 80:
            status_text = color_text(f"🟢 AGREED ({num_agree}/{num_segments} segments matched)", Colors.GREEN, color_enabled)
        elif agreement_rate >= 40:
            status_text = color_text(f"🟡 PARTIALLY AGREED ({num_agree}/{num_segments} segments matched)", Colors.YELLOW, color_enabled)
        else:
            status_text = color_text(f"🔴 DISAGREED ({num_agree}/{num_segments} segments matched)", Colors.RED, color_enabled)

        print("\n" + color_text("CROSS-CHECK SUMMARY (Dual-Path):", Colors.BOLD, color_enabled))
        print(f"  Agreement Rate:    {agreement_rate:.1f}%")
        print(f"  Verification:      {status_text}")

    # Re-segment to get segment text if detailed is True and text_to_analyze is provided
    seg_texts = []
    if detailed and text_to_analyze:
        try:
            lang = metadata.get("language", "hi")
            strategy = metadata.get("segment_strategy", "sentence")
            if lang == "en":
                from utils.preprocessing_en import preprocess_pipeline_en
                seg_texts, _ = preprocess_pipeline_en(text_to_analyze, segment_strategy=strategy)
            else:
                from utils.preprocessing import preprocess_pipeline
                seg_texts, _ = preprocess_pipeline(text_to_analyze, segment_strategy=strategy)
        except Exception:
            pass

    # Print detailed segments if requested
    if detailed and segments:
        print("\n" + color_text("DETAILED SEGMENT ANALYSIS:", Colors.BOLD + Colors.CYAN, color_enabled))
        print("=" * 72)
        for seg_idx, seg in enumerate(segments):
            seg_id = seg.get("segment_id", "seg")
            text_span = seg.get("text_span", "")
            
            span_text = ""
            if seg_texts and seg_idx < len(seg_texts):
                span_text = seg_texts[seg_idx]
            else:
                span_text = seg.get("text", "")
                if not span_text and "text_span" in seg:
                    span_text = seg["text_span"]
            
            # Truncate text span if it's too long
            span_preview = span_text[:80] + "..." if len(span_text) > 83 else span_text
            
            sentiment_dict = seg.get("sentiment", {})
            seg_label = sentiment_dict.get("label", "unknown").upper()
            seg_pol = sentiment_dict.get("polarity_score", 0.0)
            seg_confidence = sentiment_dict.get("confidence", 0.0) * 100
            
            if seg_pol > 0.05:
                seg_pol_color = Colors.GREEN
            elif seg_pol < -0.05:
                seg_pol_color = Colors.RED
            else:
                seg_pol_color = Colors.YELLOW

            print(f"\n{color_text(f'Segment {seg_id}:', Colors.BOLD, color_enabled)} \"{span_preview}\"")
            print(f"  Sentiment:         {color_text(seg_label, Colors.MAGENTA, color_enabled)} (Confidence: {seg_confidence:.1f}%)")
            print(f"  Polarity:          {color_text(f'{seg_pol:+.2f}', seg_pol_color, color_enabled)}")
            
            # Seg cross check
            seg_cc = seg.get("cross_check", {})
            if seg_cc:
                agree = seg_cc.get("agreement", True)
                agree_status = color_text("AGREE", Colors.GREEN, color_enabled) if agree else color_text("DISAGREE", Colors.RED, color_enabled)
                hindi_lbl = seg_cc.get("hindi_direct", {}).get("label", "N/A")
                english_lbl = seg_cc.get("english_translated", {}).get("label", "N/A")
                print(f"  Cross-Check:       {agree_status} (Hindi: {hindi_lbl} vs English-translated: {english_lbl})")
        print("=" * 72)
    else:
        print(f"\n💡 Run with {color_text('--detailed', Colors.BOLD, color_enabled)} flag to view per-segment sentiment analysis and translation comparisons.")
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment CLI: Dual-Path Sentiment Analysis on Hindi and English Text."
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to a .txt file to analyze (positional shortcut, e.g. python main.py doc.txt)."
    )
    parser.add_argument(
        "--text", "-t", type=str, help="Text to analyze directly from the command line."
    )
    parser.add_argument(
        "--file", "-f", type=str, help="Path to a .txt file to read and analyze."
    )
    parser.add_argument(
        "--language", "-l", choices=["hi", "en"], help="Explicitly specify language code (hi=Hindi, en=English)."
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Save the detailed analysis JSON output to this file path."
    )
    parser.add_argument(
        "--detailed", "-d", action="store_true", help="Print detailed segment-by-segment analysis."
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output in the console."
    )

    args = parser.parse_args()
    color_enabled = not args.no_color

    # Positional .txt file takes precedence over --file flag
    if args.input_file:
        if not args.input_file.endswith(".txt"):
            print(color_text(f"[!] Positional argument must be a .txt file, got: {args.input_file}", Colors.RED, color_enabled))
            sys.exit(1)
        args.file = args.input_file

    print_banner(color_enabled)

    # Load model
    print(color_text("[~] Loading models and pipeline (this may take 15-30 seconds)...", Colors.YELLOW, color_enabled))
    logger.warning("Initializing RealTimeInference pipeline...")
    try:
        inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)
        device_name = str(inference.pipeline.device)
        print(color_text(f"[+] Pipeline successfully loaded on device: {device_name}", Colors.GREEN, color_enabled))
    except Exception as e:
        print(color_text(f"[!] Failed to load model pipeline: {str(e)}", Colors.RED, color_enabled))
        sys.exit(1)

    # Process single-shot text
    if args.text:
        print(f"\nAnalyzing text input...")
        result = inference.predict(args.text, language=args.language)
        format_sentiment_result(result, text_to_analyze=args.text, detailed=args.detailed, color_enabled=color_enabled)
        
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(color_text(f"\n[+] Detailed JSON results saved to: {args.output}", Colors.GREEN, color_enabled))
            except Exception as e:
                print(color_text(f"\n[!] Error saving JSON results: {str(e)}", Colors.RED, color_enabled))
        return

    # Process file input
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(color_text(f"[!] Error: File not found at {args.file}", Colors.RED, color_enabled))
            sys.exit(1)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                print(color_text("[!] Error: File is empty.", Colors.RED, color_enabled))
                sys.exit(1)

            print(f"\nAnalyzing file content ({file_path.name})...")
            result = inference.predict(content, language=args.language)
            format_sentiment_result(result, text_to_analyze=content, detailed=args.detailed, color_enabled=color_enabled)

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(color_text(f"\n[+] Detailed JSON results saved to: {args.output}", Colors.GREEN, color_enabled))

        except Exception as e:
            print(color_text(f"[!] Error reading/analyzing file: {str(e)}", Colors.RED, color_enabled))
            sys.exit(1)
        return

    # No single-shot arguments; launch interactive mode
    print("\n" + color_text("--- Interactive CLI Mode ---", Colors.BOLD + Colors.BLUE, color_enabled))
    print("Commands:")
    print(f"  • {color_text('exit', Colors.BOLD, color_enabled)} / {color_text('quit', Colors.BOLD, color_enabled)} : Terminate program")
    print(f"  • {color_text('file <path>', Colors.BOLD, color_enabled)} : Analyze a text file")
    print(f"  • {color_text('multiline', Colors.BOLD, color_enabled)}   : Enter multi-line input mode (end with an empty line)")
    print(f"  • {color_text('detailed', Colors.BOLD, color_enabled)}    : Toggle segment-level details (Currently: {'ON' if args.detailed else 'OFF'})")
    print(f"  • {color_text('help', Colors.BOLD, color_enabled)}        : Print this information again\n")

    detailed_mode = args.detailed

    while True:
        try:
            prompt = color_text("senti-cli> ", Colors.BOLD + Colors.CYAN, color_enabled)
            user_input = input(prompt).strip()
            
            if not user_input:
                continue

            # Command routing
            lower_input = user_input.lower()
            if lower_input in ["exit", "quit"]:
                print(color_text("Goodbye!", Colors.GREEN, color_enabled))
                break
            
            elif lower_input == "help":
                print("\nCommands:")
                print(f"  • {color_text('exit', Colors.BOLD, color_enabled)} / {color_text('quit', Colors.BOLD, color_enabled)} : Terminate program")
                print(f"  • {color_text('file <path>', Colors.BOLD, color_enabled)} : Analyze a text file")
                print(f"  • {color_text('multiline', Colors.BOLD, color_enabled)}   : Enter multi-line input mode (end with an empty line)")
                print(f"  • {color_text('detailed', Colors.BOLD, color_enabled)}    : Toggle segment-level details (Currently: {'ON' if detailed_mode else 'OFF'})")
                print(f"  • {color_text('help', Colors.BOLD, color_enabled)}        : Print this information again\n")
                continue

            elif lower_input == "detailed":
                detailed_mode = not detailed_mode
                print(f"Per-segment detailed view: {color_text('ON' if detailed_mode else 'OFF', Colors.BOLD, color_enabled)}")
                continue

            elif lower_input.startswith("file "):
                file_path_str = user_input[5:].strip()
                file_path = Path(file_path_str)
                if not file_path.exists():
                    print(color_text(f"[!] Error: File not found at {file_path_str}", Colors.RED, color_enabled))
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_to_analyze = f.read().strip()
                    if not text_to_analyze:
                        print(color_text("[!] Error: File is empty.", Colors.RED, color_enabled))
                        continue
                    print(f"Analyzing file: {file_path.name}...")
                except Exception as e:
                    print(color_text(f"[!] Error reading file: {str(e)}", Colors.RED, color_enabled))
                    continue

            elif lower_input == "multiline":
                print(color_text("Entering multi-line mode. Type your text and press Enter on an empty line to analyze:", Colors.YELLOW, color_enabled))
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                text_to_analyze = "\n".join(lines).strip()
                if not text_to_analyze:
                    print("[!] No input provided.")
                    continue
                print("Analyzing multi-line input...")

            else:
                text_to_analyze = user_input

            # Run inference
            try:
                result = inference.predict(text_to_analyze)
                format_sentiment_result(result, text_to_analyze=text_to_analyze, detailed=detailed_mode, color_enabled=color_enabled)
            except Exception as e:
                print(color_text(f"[!] Error running analysis: {str(e)}", Colors.RED, color_enabled))

        except KeyboardInterrupt:
            print("\n" + color_text("Goodbye!", Colors.GREEN, color_enabled))
            break
        except EOFError:
            print("\n" + color_text("Goodbye!", Colors.GREEN, color_enabled))
            break


if __name__ == "__main__":
    main()