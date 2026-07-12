"""
Configuration file for Bilingual Text Summarizer
Contains model parameters, prompts, and optimization settings
"""

import os
from pathlib import Path
from typing import Dict

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Available models (ranked by performance vs speed)
AVAILABLE_MODELS = {
    'llama-3.2-3b': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'description': 'Balanced model - good quality and speed',
        'params': '3B',
        'min_memory_gb': 8,
        'context_window': 8192,
        'default': True
    },
    'llama-3.2-1b': {
        'name': 'meta-llama/Llama-3.2-1B-Instruct',
        'description': 'Lightweight model - faster inference',
        'params': '1B',
        'min_memory_gb': 4,
        'context_window': 8192,
        'default': False
    },
}

# Default model
DEFAULT_MODEL = 'llama-3.2-3b'

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

# Summary length configurations
SUMMARY_LENGTHS = {
    'short': {
        'min_tokens': 50,
        'max_tokens': 150,
        'target_sentences': '2-3'
    },
    'medium': {
        'min_tokens': 150,
        'max_tokens': 300,
        'target_sentences': '4-6'
    },
    'long': {
        'min_tokens': 300,
        'max_tokens': 512,
        'target_sentences': '8-10'
    }
}

# Generation parameters for model inference
GENERATION_CONFIG = {
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.0,
    'do_sample': True,
    'num_beams': 1,  # Set to 1 for faster generation
}

# Hierarchical summarization chunk size (in tokens)
CHUNK_SIZE = 2000
HIERARCHICAL_THRESHOLD = 2000  # Use hierarchical approach for texts > this size

# ============================================================================
# LANGUAGE-SPECIFIC CONFIGURATIONS
# ============================================================================

# Language detection thresholds
LANGUAGE_CONFIG = {
    'hindi': {
        'code': 'hi',
        'script': 'Devanagari',
        'unicode_start': 0x0900,
        'unicode_end': 0x097F,
        'detection_threshold': 0.3,  # 30% of characters must be Devanagari
        'iso_639_1': 'hi',
        'iso_639_3': 'hin'
    },
    'english': {
        'code': 'en',
        'script': 'Latin',
        'unicode_start': 0x0041,
        'unicode_end': 0x007A,
        'detection_threshold': 0.0,
        'iso_639_1': 'en',
        'iso_639_3': 'eng'
    }
}

# ============================================================================
# PROMPTS - ENGLISH
# ============================================================================

ENGLISH_PROMPTS = {
    'short': """Summarize the following text in 2-3 sentences. Focus on the main idea and key points:

{text}

Summary:""",
    
    'medium': """Provide a concise summary of the following text in 4-6 sentences. Include the main idea, key points, and important details:

{text}

Summary:""",
    
    'long': """Provide a detailed summary of the following text in 8-10 sentences. Include the main idea, key arguments, important examples, and conclusions:

{text}

Summary:"""
}

# ============================================================================
# PROMPTS - HINDI
# ============================================================================

HINDI_PROMPTS = {
    'short': """निम्नलिखित पाठ को 2-3 वाक्यों में सारांश दें। मुख्य विचार और महत्वपूर्ण बिंदुओं पर ध्यान दें:

{text}

सारांश:""",
    
    'medium': """निम्नलिखित पाठ का 4-6 वाक्यों में संक्षिप्त सारांश दें। मुख्य विचार, महत्वपूर्ण बिंदु और विवरण शामिल करें:

{text}

सारांश:""",
    
    'long': """निम्नलिखित पाठ का 8-10 वाक्यों में विस्तृत सारांश दें। मुख्य विचार, महत्वपूर्ण तर्क, उदाहरण और निष्कर्ष शामिल करें:

{text}

सारांश:"""
}

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

# Quantization settings
QUANTIZATION_CONFIG = {
    'enabled': False,
    '4bit': {
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_compute_dtype': 'float16'
    },
    '8bit': {
        'load_in_8bit': True,
    }
}

# Cache settings
CACHE_CONFIG = {
    'use_cache': True,
    'cache_dir': str(Path.home() / '.cache' / 'huggingface' / 'hub'),
    'offload_to_cpu': False
}

# ============================================================================
# FILE CONFIGURATION
# ============================================================================

# Supported file types
SUPPORTED_FORMATS = {
    'text': ['.txt'],
    'pdf': ['.pdf'],  # Future support
    'documents': ['.docx', '.doc'],  # Future support
}

# Default encoding
FILE_ENCODING = 'utf-8'

# Maximum file size (in MB)
MAX_FILE_SIZE_MB = 100

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

LOG_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'summarizer.log',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s'
}

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': '📚 Bilingual Text Summarizer',
    'page_icon': '📚',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_upload_size_mb': 50,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_config(model_name: str = DEFAULT_MODEL) -> Dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Dictionary containing model configuration
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    return AVAILABLE_MODELS[model_name]


def get_prompts(language: str = 'en') -> Dict[str, str]:
    """
    Get prompts for a specific language.
    
    Args:
        language: Language code ('en' or 'hi')
        
    Returns:
        Dictionary of prompts for different summary lengths
    """
    if language == 'en':
        return ENGLISH_PROMPTS
    elif language == 'hi':
        return HINDI_PROMPTS
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_language_config(language: str) -> Dict:
    """
    Get language-specific configuration.
    
    Args:
        language: Language code ('en' or 'hi')
        
    Returns:
        Dictionary containing language configuration
    """
    for lang_name, config in LANGUAGE_CONFIG.items():
        if config['code'] == language:
            return config
    
    raise ValueError(f"Unknown language: {language}")


def get_generation_config(summary_length: str = 'medium') -> Dict:
    """
    Get generation configuration adjusted for summary length.
    
    Args:
        summary_length: 'short', 'medium', or 'long'
        
    Returns:
        Dictionary containing generation parameters
    """
    config = GENERATION_CONFIG.copy()
    
    if summary_length in SUMMARY_LENGTHS:
        config['max_new_tokens'] = SUMMARY_LENGTHS[summary_length]['max_tokens']
    
    return config


def validate_file(file_path: str) -> bool:
    """
    Validate if file is supported and accessible.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, raises exception otherwise
    """
    path = Path(file_path)
    
    # Check if exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.2f}MB (max: {MAX_FILE_SIZE_MB}MB)")
    
    # Check file type
    if path.suffix.lower() not in SUPPORTED_FORMATS['text']:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return True


def validate_text(text: str) -> bool:
    """
    Validate input text.
    
    Args:
        text: Input text
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) < 10:
        raise ValueError("Text must be at least 10 characters")
    
    return True


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

class EnvConfig:
    """Environment-based configuration"""
    
    @staticmethod
    def get_model_name() -> str:
        """Get model from environment or use default"""
        return os.getenv('SUMMARIZER_MODEL', DEFAULT_MODEL)
    
    @staticmethod
    def get_cache_dir() -> str:
        """Get cache directory from environment or use default"""
        return os.getenv('HF_HOME', CACHE_CONFIG['cache_dir'])
    
    @staticmethod
    def get_device() -> str:
        """Get device preference from environment"""
        device = os.getenv('SUMMARIZER_DEVICE', 'auto')
        if device not in ['auto', 'cuda', 'cpu']:
            device = 'auto'
        return device
    
    @staticmethod
    def use_quantization() -> bool:
        """Check if quantization is enabled via environment"""
        return os.getenv('SUMMARIZER_QUANTIZE', 'false').lower() == 'true'


# ============================================================================
# PRINT CONFIGURATION (for debugging)
# ============================================================================

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("SUMMARIZER CONFIGURATION")
    print("="*60)
    
    print("\nMODEL:")
    model_config = get_model_config(DEFAULT_MODEL)
    print(f"  Name: {model_config['name']}")
    print(f"  Params: {model_config['params']}")
    print(f"  Context: {model_config['context_window']} tokens")
    print(f"  Min Memory: {model_config['min_memory_gb']}GB")
    
    print("\nGENERATION:")
    print(f"  Temperature: {GENERATION_CONFIG['temperature']}")
    print(f"  Top-P: {GENERATION_CONFIG['top_p']}")
    print(f"  Max Tokens: {GENERATION_CONFIG['max_new_tokens']}")
    
    print("\nLANGUAGES:")
    for lang_name, config in LANGUAGE_CONFIG.items():
        print(f"  {lang_name.upper()}: {config['code']} ({config['script']})")
    
    print("\nCHUNKING:")
    print(f"  Chunk Size: {CHUNK_SIZE} tokens")
    print(f"  Hierarchical Threshold: {HIERARCHICAL_THRESHOLD} tokens")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()