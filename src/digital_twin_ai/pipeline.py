"""
pipeline.py - Main entrypoint for the AI pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .clustering import ClusteringManager
from .config import PipelineConfig
from .data_generation import DataGenerator
from .feature_engineering import FeatureEngineer
from .persona_modeling import PersonaManager


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full AI pipeline:
    1. Data Generation (Dummy if needed, then Synthetic)
    2. Feature Engineering
    3. Clustering
    4. Persona Modeling
    """
    pipeline_config = PipelineConfig.model_validate(config)
    print("Starting AI Pipeline...")

    output_dir = pipeline_config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    dg = DataGenerator(random_state=pipeline_config.random_state)

    if not pipeline_config.excel_file.exists():
        pipeline_config.excel_file.parent.mkdir(parents=True, exist_ok=True)
        dg.generate_dummy_excel(2000, str(pipeline_config.excel_file))

    distributions = dg.load_real_distributions(str(pipeline_config.excel_file))
    synthetic_df = dg.generate_synthetic_data(
        pipeline_config.n_synthetic_customers,
        distributions,
    )

    synthetic_path = output_dir / "synthetic_customers.parquet"
    synthetic_df.to_parquet(synthetic_path, index=False)
    print(f"Synthetic data saved to {synthetic_path}")

    fe = FeatureEngineer()
    features_df, feature_cols = fe.process_features(synthetic_df)

    features_path = output_dir / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    print(f"Features saved to {features_path}")

    cm = ClusteringManager(
        n_personas=pipeline_config.n_personas,
        random_state=pipeline_config.random_state,
    )
    embedding = cm.reduce_dimensions(features_df, feature_cols)
    labels = cm.cluster(embedding)

    cluster_plot_path = output_dir / "clusters_umap.png"
    cm.visualize_clusters(embedding, labels, str(cluster_plot_path))

    features_df["persona_cluster"] = labels
    features_df["umap_x"] = embedding[:, 0]
    features_df["umap_y"] = embedding[:, 1]

    clustered_path = output_dir / "clustered_customers.parquet"
    features_df.to_parquet(clustered_path, index=False)
    print(f"Clustered data saved to {clustered_path}")

    pm = PersonaManager(
        api_key=pipeline_config.gemini_api_key,
        model_name=pipeline_config.gemini_model_name,
    )

    personas = []
    cluster_ids = sorted(features_df["persona_cluster"].unique())
    for cluster_id in cluster_ids:
        print(f"Processing cluster {cluster_id}...")
        stats = pm.extract_cluster_stats(features_df, cluster_id)
        persona = pm.generate_persona_profile(stats)
        
        # 클러스터 내 샘플을 추출하여 개별 페르소나 스토리 생성 추가
        cluster_df = features_df[features_df["persona_cluster"] == cluster_id]
        sample_df = cluster_df.sample(min(3, len(cluster_df)), random_state=pipeline_config.random_state)
        persona["individual_stories"] = pm.generate_individual_stories(sample_df)
        
        personas.append(persona)

    personas_path = output_dir / "personas.json"
    personas_path.write_text(
        json.dumps(personas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Personas saved to {personas_path}")

    metadata = {
        "status": "success",
        "random_state": pipeline_config.random_state,
        "n_synthetic_customers": pipeline_config.n_synthetic_customers,
        "n_personas_requested": pipeline_config.n_personas,
        "n_personas_generated": len(personas),
        "excel_path": str(pipeline_config.excel_file),
        "outputs": {
            "synthetic_data": str(synthetic_path),
            "features": str(features_path),
            "clustered_data": str(clustered_path),
            "personas": str(personas_path),
            "cluster_plot": str(cluster_plot_path),
        },
    }
    metadata_path = output_dir / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return metadata


def run_eda(excel_path: str, output_dir: str) -> None:
    """Simple EDA summary similar to 01_eda.py."""
    print(f"Running EDA on {excel_path}...")
    sheets = ["Demo", "구매", "보유", "앱사용", "관심사", "CLV", "리워즈"]
    data = {sheet: pd.read_excel(excel_path, sheet_name=sheet, header=1) for sheet in sheets}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "eda_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "=== EDA Summary ===",
                f"Total customers: {data['Demo']['index'].nunique()}",
                f"Purchase count: {len(data['구매'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"EDA summary saved to {summary_path}")
