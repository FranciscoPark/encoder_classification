import os
import json
import math

# Robust NaN checker
def is_nan(x):
    return (
        x is None or
        x == "" or
        x == "NaN" or
        x == 0 or
        (isinstance(x, float) and math.isnan(x))
    )

#input_dir = "/home/s1/jypark/encoder_classification/trainset"
#output_file = "/home/s1/jypark/encoder_classification/trainset/merged_cleaned.json"
input_dir ="/home/s1/jypark/encoder_classification/validationset"
output_file = "/home/s1/jypark/encoder_classification/validationset/merged_cleaned.json"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

all_cleaned = []
total = 0

json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

for fname in sorted(json_files):
    input_path = os.path.join(input_dir, fname)

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data_list = json.load(f)
        except Exception as e:
            print(f"❌ Skipping {fname}: {e}")
            continue

    is_special_task = "hellaswag" in fname.lower() or "winogrande" in fname.lower()

    file_cleaned = []
    for item in data_list:
        case_1 = item.get("case_1")
        case = item.get("case")

        # Skip if both are missing/invalid
        if is_nan(case_1) and is_nan(case):
            continue

        cleaned = {
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer"),
        }

        if is_special_task:
            cleaned["case"] = case
        elif not is_special_task:
            cleaned["case"] = case_1
        else:
            continue  # Just in case both are invalid (double check)

        file_cleaned.append(cleaned)

    all_cleaned.extend(file_cleaned)
    print(f"✅ {fname}: {len(file_cleaned)} examples kept")

# Save merged result
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_cleaned, f, ensure_ascii=False, indent=2)

print(f"\n✅ Total merged examples: {len(all_cleaned)}")
print(f"✅ Saved to: {output_file}")