import argparse
import os
import sys
from huggingface_hub import hf_hub_download


MODELS_DIR = "models"

MODELS = {
    "gemma-3-12b-q5": {
        "repo": os.environ.get("ODBCHAT_GEMMA_12B_REPO", "bartowski/google_gemma-3-12b-it-GGUF"),
        "files": ["google_gemma-3-12b-it-Q5_K_M.gguf"],
    },
    "gemma-3-1b-q4": {
        "repo": os.environ.get("ODBCHAT_GEMMA_1B_REPO", "bartowski/google_gemma-3-1b-it-GGUF"),
        "files": ["google_gemma-3-1b-it-Q4_K_M.gguf"],
    },
}


def download_model(model_key: str) -> None:
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    config = MODELS[model_key]
    repo = config["repo"]
    print(f"Targeting model: {model_key} from {repo}")

    for filename in config["files"]:
        print(f"Downloading {filename}...")
        try:
            file_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded: {file_path}")
        except Exception as exc:
            print(f"Error downloading {filename}: {exc}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF models for odbchat")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-12b-q5",
        choices=MODELS.keys(),
        help="Which model to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all configured models",
    )
    args = parser.parse_args()

    if args.all:
        for key in MODELS:
            download_model(key)
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()
