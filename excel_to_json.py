import pandas as pd
import json
import os

# Input Excel file (with multiple sheets)
#input_file = "/home/s1/jypark/encoder_classification/dataset/train.xlsx" 
#output_dir = "json_outputs"   
input_file = "/home/s1/jypark/encoder_classification/dataset/validation.xlsx"
output_dir =  "json_validation"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all sheets
sheets = pd.read_excel(input_file, sheet_name=None)  

# Iterate over each sheet
for sheet_name, df in sheets.items():
    # Convert to list of dicts
    json_data = df.to_dict(orient="records")

    # Define output path
    output_path = os.path.join(output_dir, f"{sheet_name}.json")

    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")