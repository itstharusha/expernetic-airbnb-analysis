"""
add_save_cell.py  — adds model-serialisation cell to 02_price_prediction.ipynb
Run once from project root: python add_save_cell.py
"""

import json

NB_PATH = "notebooks/02_price_prediction.ipynb"

save_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "save_model_cell",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Save Model Artefacts ────────────────────────────────────────────────\n",
        "# Serialise the final XGBoost model, SHAP explainer, and all metadata\n",
        "# so the Streamlit dashboard can load them without re-training.\n",
        "import os, joblib\n",
        "\n",
        'MODELS_DIR = "../models"\n',
        "os.makedirs(MODELS_DIR, exist_ok=True)\n",
        "\n",
        "# Final model (trained on full dataset in the cell above)\n",
        'joblib.dump(final_model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))\n',
        "\n",
        "# SHAP explainer\n",
        'joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))\n',
        "\n",
        "# Imputation medians (computed on the training dataframe `df`)\n",
        "IMPUTE_COLS = ['bathrooms', 'bedrooms', 'beds', 'host_listings_count',\n",
        "               'hosts_time_as_host_years', 'minimum_nights']\n",
        "medians = {col: float(df[col].median()) for col in IMPUTE_COLS}\n",
        "\n",
        "# Metadata bundle\n",
        "meta = {\n",
        "    'feature_cols':  feature_cols,\n",
        "    'ng_dummy_cols': list(neighbourhood_dummies.columns),\n",
        "    'medians':       medians,\n",
        "    'room_type_enc': {\n",
        "        'Entire home/apt': 3,\n",
        "        'Private room':    2,\n",
        "        'Hotel room':      1,\n",
        "        'Shared room':     0,\n",
        "    },\n",
        "}\n",
        'joblib.dump(meta, os.path.join(MODELS_DIR, "model_meta.joblib"))\n',
        "\n",
        "print('✅  Model artefacts saved:')\n",
        "for f in sorted(os.listdir(MODELS_DIR)):\n",
        "    size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024\n",
        "    print(f'   {f:<35}  {size_kb:>5} KB')\n",
    ],
}

md_cell = {
    "cell_type": "markdown",
    "id": "save_model_md",
    "metadata": {},
    "source": [
        "## Save Model Artefacts\n",
        "\n",
        "Serialise the fitted XGBoost model, its SHAP explainer, and all feature-engineering\n",
        "metadata (column list, imputation medians, encoding maps) to `models/` so the\n",
        "Streamlit dashboard can load them without re-training.",
    ],
}

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove any previous save-cell (idempotent)
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in ("save_model_cell", "save_model_md")]

nb["cells"].append(md_cell)
nb["cells"].append(save_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — appended 2 cells to {NB_PATH}")
