# Configuration
MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024

# Summary length options
SUMMARY_OPTIONS = {
    "1": {
        "name": "Short",
        "max_length": 60,
        "min_length": 20
    },
    "2": {
        "name": "Medium",
        "max_length": 130,
        "min_length": 40
    },
    "3": {
        "name": "Long",
        "max_length": 200,
        "min_length": 80
    }
}


# Terminal colors
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


# Display formatted messages
def print_message(message, color=Colors.RESET):
    print(f"{color}{message}{Colors.RESET}")


# Display section headings
def print_header(title):
    print("\n" + "=" * 60)
    print_message(title.center(60), Colors.BOLD + Colors.CYAN)
    print("=" * 60)
