from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    random_state: int = 42
    n_synthetic_customers: int = Field(default=1000, ge=100)
    n_personas: int = Field(default=7, ge=2, le=20)
    excel_path: str
    output_dir: str = "./output"
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3.0-flash"

    @field_validator("excel_path", "output_dir")
    @classmethod
    def validate_non_empty_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Path values must not be empty.")
        return value

    @property
    def excel_file(self) -> Path:
        return Path(self.excel_path)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
