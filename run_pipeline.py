import os

from digital_twin_ai import run_pipeline


def main() -> None:
    config = {
        "random_state": int(os.getenv("RANDOM_STATE", "42")),
        "n_synthetic_customers": int(os.getenv("SYNTHETIC_N_CUSTOMERS", "1000")),
        "n_personas": int(os.getenv("N_PERSONAS", "7")),
        "excel_path": os.getenv("EXCEL_PATH", "./data/Digital Customer Twin.xlsx"),
        "output_dir": os.getenv("OUTPUT_DIR", "./output"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model_name": os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash"),
    }
    result = run_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
