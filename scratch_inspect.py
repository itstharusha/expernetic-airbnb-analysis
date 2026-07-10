import json

def inspect_notebook(path, out_f):
    out_f.write(f"\n==================== Inspecting {path} ====================\n")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        source_str = "".join(source).strip()
        first_line = source[0].strip() if source else ""
        
        if cell_type == 'markdown':
            out_f.write(f"Cell {idx} [Markdown]: {source_str[:150]}...\n")
        elif cell_type == 'code':
            out_f.write(f"Cell {idx} [Code]: {first_line[:80]}...\n")

if __name__ == "__main__":
    with open("notebook_structure.txt", "w", encoding="utf-8") as out_f:
        inspect_notebook("notebooks/01_eda.ipynb", out_f)
        inspect_notebook("notebooks/02_price_prediction.ipynb", out_f)
        inspect_notebook("notebooks/03_nlp_reviews.ipynb", out_f)
        inspect_notebook("notebooks/04_recommender.ipynb", out_f)
        inspect_notebook("notebooks/05_generative_ai.ipynb", out_f)
