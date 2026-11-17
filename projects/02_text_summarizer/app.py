python

```python

from dataclasses import dataclass
from typing import Dict, Optional
import os
import sys

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024

GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "length_penalty": 2.0,
    "truncation": True
}

SUMMARY_OPTIONS = {
    "1": {
        "name": "Short",
        "max_length": 60,
        "min_length": 20,
        "description": "Quick and concise summary"
    },
    "2": {
        "name": "Medium",
        "max_length": 130,
        "min_length": 40,
        "description": "Balanced and informative summary"
    },
    "3": {
        "name": "Long",
        "max_length": 200,
        "min_length": 80,
        "description": "Detailed and comprehensive summary"
    }
}

class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def print_message(
    message: str,
    color: str = Colors.RESET,
    bold: bool = False
) -> None:
    style = Colors.BOLD if bold else ""
    print(f"{style}{color}{message}{Colors.RESET}")


def print_header(title: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print_message(title.center(width), Colors.CYAN, bold=True)
    print("=" * width)


def print_success(message: str) -> None:
    print_message(f"[SUCCESS] {message}", Colors.GREEN)


def print_error(message: str) -> None:
    print_message(f"[ERROR] {message}", Colors.RED)


def print_warning(message: str) -> None:
    print_message(f"[WARNING] {message}", Colors.YELLOW)


def print_info(message: str) -> None:
    print_message(f"[INFO] {message}", Colors.BLUE)


def display_summary_options() -> None:
    print_header("SUMMARY LENGTH OPTIONS")

    for key, option in SUMMARY_OPTIONS.items():
        print(
            f"{Colors.GREEN}{key}.{Colors.RESET} "
            f"{Colors.BOLD}{option['name']}{Colors.RESET}"
        )
        print(f"   {option['description']}")
        print(
            f"   Length: {option['min_length']}-"
            f"{option['max_length']} tokens\n"
        )


def get_summary_config(choice: str) -> dict:
    if choice not in SUMMARY_OPTIONS:
        raise ValueError(
            f"Invalid choice '{choice}'. Select 1, 2, or 3."
        )

    return SUMMARY_OPTIONS[choice]


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    return text


def validate_summary_choice(choice: str) -> bool:
    return choice in SUMMARY_OPTIONS


def display_application_info() -> None:
    print_header("AI TEXT SUMMARIZATION SYSTEM")

    print_message("Model: " + MODEL_NAME, Colors.CYAN)
    print_message(
        f"Maximum Input Tokens: {MAX_INPUT_TOKENS}",
        Colors.CYAN
    )
    print_message(
        "Framework: Hugging Face Transformers",
        Colors.CYAN
    )
    print_message(
        "Status: Ready for summarization",
        Colors.GREEN
    )


def configure_terminal() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if os.name == "nt":
        os.system("")


def main() -> None:
    configure_terminal()
    display_application_info()
    display_summary_options()
    print_success("Configuration loaded successfully.")


if __name__ == "__main__":
    main()
```
