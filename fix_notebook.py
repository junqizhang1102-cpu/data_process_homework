import json
import os

NOTEBOOK_PATH = 'd:\\2_School\\204_Course Information\\dataprocess\\notebook303e42d520.ipynb'

# Keywords to identify the cells
LLM_RESID_KEYWORD = "LLM_resid"
RESIDUAL_PLOT_KEYWORD = "Residual Comparison by Model"

print(f"Processing notebook: {NOTEBOOK_PATH}")

try:
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    llm_resid_cell_index = -1
    residual_plot_cell_index = -1

    # Find the indices of the cells to move
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        
        cell_source = ''.join(cell.get('source', []))
        
        if LLM_RESID_KEYWORD in cell_source:
            llm_resid_cell_index = i
            print(f"Found LLM residual calculation cell at index: {i}")

        if RESIDUAL_PLOT_KEYWORD in cell_source:
            residual_plot_cell_index = i
            print(f"Found residual plot cell at index: {i}")

    # If both cells are found and the resid cell is after the plot cell, move it
    if llm_resid_cell_index != -1 and residual_plot_cell_index != -1:
        if llm_resid_cell_index \u003e residual_plot_cell_index:
            print(f"Moving cell {llm_resid_cell_index} to before cell {residual_plot_cell_index}...")
            resid_cell = nb['cells'].pop(llm_resid_cell_index)
            nb['cells'].insert(residual_plot_cell_index, resid_cell)

            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print("Notebook updated successfully.")
        else:
            print("No changes needed, cells are already in the correct order.")
    else:
        if llm_resid_cell_index == -1:
            print("Warning: LLM residual calculation cell not found.")
        if residual_plot_cell_index == -1:
            print("Warning: Residual plot cell not found.")

except FileNotFoundError:
    print(f"Error: Notebook file not found at {NOTEBOOK_PATH}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {NOTEBOOK_PATH}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")