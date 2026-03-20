"""
pipeline.py - Main entrypoint for the AI pipeline.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .data_generation import DataGenerator
from .feature_engineering import FeatureEngineer
from .clustering import ClusteringManager
from .persona_modeling import PersonaManager


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full AI pipeline:
    1. Data Generation (Dummy if needed, then Synthetic)
    2. Feature Engineering
    3. Clustering
    4. Persona Modeling
    """
    print("Starting AI Pipeline...")
    
    # Extract config
    random_state = config.get("random_state", 42)
    n_synthetic = config.get("n_synthetic_customers", 1000)
    n_personas = config.get("n_personas", 7)
    excel_path = config.get("excel_path")
    output_dir = config.get("output_dir", "./output")
    gemini_api_key = config.get("gemini_api_key")
    gemini_model = config.get("gemini_model_name", "gemini-3-flash")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data Generation
    dg = DataGenerator(random_state=random_state)
    
    # If excel doesn't exist, generate dummy excel
    if excel_path and not os.path.exists(excel_path):
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        dg.generate_dummy_excel(2000, excel_path)
    
    # Load distributions and generate synthetic data
    if excel_path:
        dist = dg.load_real_distributions(excel_path)
        synthetic_df = dg.generate_synthetic_data(n_synthetic, dist)
    else:
        raise ValueError("excel_path is required for generating synthetic data.")
    
    synthetic_path = os.path.join(output_dir, "synthetic_customers.parquet")
    synthetic_df.to_parquet(synthetic_path, index=False)
    print(f"Synthetic data saved to {synthetic_path}")

    # 2. Feature Engineering
    fe = FeatureEngineer()
    features_df, feature_cols = fe.process_features(synthetic_df)
    
    features_path = os.path.join(output_dir, "features.parquet")
    features_df.to_parquet(features_path, index=False)
    print(f"Features saved to {features_path}")

    # 3. Clustering
    cm = ClusteringManager(n_personas=n_personas, random_state=random_state)
    embedding = cm.reduce_dimensions(features_df, feature_cols)
    labels = cm.cluster(embedding)
    
    cluster_plot_path = os.path.join(output_dir, "clusters_umap.png")
    cm.visualize_clusters(embedding, labels, cluster_plot_path)
    
    features_df["persona_cluster"] = labels
    features_df["umap_x"] = embedding[:, 0]
    features_df["umap_y"] = embedding[:, 1]
    
    clustered_path = os.path.join(output_dir, "clustered_customers.parquet")
    features_df.to_parquet(clustered_path, index=False)
    print(f"Clustered data saved to {clustered_path}")

    # 4. Persona Modeling
    pm = PersonaManager(api_key=gemini_api_key, model_name=gemini_model)
    
    cluster_ids = sorted(features_df["persona_cluster"].unique())
    personas = []
    
    for cid in cluster_ids:
        print(f"Processing cluster {cid}...")
        stats = pm.extract_cluster_stats(features_df, cid)
        persona = pm.generate_persona_profile(stats)
        personas.append(persona)
        
    personas_path = os.path.join(output_dir, "personas.json")
    with open(personas_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"Personas saved to {personas_path}")

    return {
        "status": "success",
        "synthetic_data": synthetic_path,
        "clustered_data": clustered_path,
        "personas": personas_path,
        "cluster_plot": cluster_plot_path,
        "n_personas": len(personas)
    }


def run_eda(excel_path: str, output_dir: str) -> None:
    """Simple EDA summary similar to 01_eda.py."""
    print(f"Running EDA on {excel_path}...")
    sheets = ["Demo", "구매", "보유", "앱사용", "관심사", "CLV", "리워즈"]
    data = {s: pd.read_excel(excel_path, sheet_name=s, header=1) for s in sheets}
    
    summary_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== EDA Summary ===\n")
        f.write(f"Total customers: {data['Demo']['index'].nunique()}\n")
        f.write(f"Purchase count: {len(data['구매'])}\n")
        # Add more summary logic as needed
        
    print(f"EDA summary saved to {summary_path}")
