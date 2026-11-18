# Import required libraries
from dataclasses import dataclass
from typing import Dict, Optional
import os
import sys

# Model configuration
MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024

# Settings used when generating summaries
GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "length_penalty": 2.0,
    "truncation": True
}

# Available summary lengths
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


# Define terminal colors
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


# Print a message with color and optional bold text
def print_message(
    message: str,
    color: str = Colors.RESET,
    bold: bool = False
) -> None:
    style = Colors.BOLD if bold else ""
    print(f"{style}{color}{message}{Colors.RESET}")


# Display a centered title
def print_header(title: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print_message(title.center(width), Colors.CYAN, bold=True)
    print("=" * width)


# Display a success message
def print_success(message: str) -> None:
    print_message(f"[SUCCESS] {message}", Colors.GREEN)


# Display an error message
def print_error(message: str) -> None:
    print_message(f"[ERROR] {message}", Colors.RED)


# Display a warning message
def print_warning(message: str) -> None:
    print_message(f"[WARNING] {message}", Colors.YELLOW)


# Display an information message
def print_info(message: str) -> None:
    print_message(f"[INFO] {message}", Colors.BLUE)


# Display the available summary length options
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


# Get the selected summary configuration
def get_summary_config(choice: str) -> dict:
    if choice not in SUMMARY_OPTIONS:
        raise ValueError(
            f"Invalid choice '{choice}'. Select 1, 2, or 3."
        )

    return SUMMARY_OPTIONS[choice]


# Check whether the input text is valid
def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    return text


# Check whether the selected summary option is valid
def validate_summary_choice(choice: str) -> bool:
    return choice in SUMMARY_OPTIONS


# Display basic application information
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


# Configure the terminal for colored output
def configure_terminal() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Enable ANSI colors on Windows
    if os.name == "nt":
        os.system("")


# Main function to run the application
def main() -> None:
    configure_terminal()

    # Display application information
    display_application_info()

    # Show available summary options
    display_summary_options()

    # Confirm that configuration is loaded
    print_success("Configuration loaded successfully.")


# Run the application
if __name__ == "__main__":
    main()
