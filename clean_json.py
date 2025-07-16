import os
import json

# input_dir = "/home/s1/jypark/encoder_classification/json_outputs"
# output_dir = "/home/s1/jypark/encoder_classification/trainset"
input_dir = "/home/s1/jypark/encoder_classification/json_validation"
output_dir = "/home/s1/jypark/encoder_classification/validationset"
os.makedirs(output_dir, exist_ok=True)

json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

for fname in json_files:
    input_path = os.path.join(input_dir, fname)

    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # Now this is a list of items

    cleaned_list = []
    for item in data_list:
        cleaned = {
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer"),
            "case_1": item.get("case_1"),
            "case": item.get("case")
        }
        cleaned_list.append(cleaned)

    # Save cleaned list to new file
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_list, f, ensure_ascii=False, indent=2)

    print(f"Processed: {fname}")