import json
import random
from collections import Counter

def process_and_save_dataset(
    #input_path="/home/s1/jypark/encoder_classification/trainset/merged_cleaned.json",
    #output_path="/home/s1/jypark/encoder_classification/trainset/processed.json",
    input_path="/home/s1/jypark/encoder_classification/validationset/merged_cleaned.json",
    output_path="/home/s1/jypark/encoder_classification/validationset/processed.json",
    seed=42,
):
    # Mapping from raw case → target label
    case_map = {
        "special": "special",
        "log": "log",
        "logc": "log",
        "which": "single",
        "word": "single",
        "i": "single",
        "d": "log",
        "this": "single",
        "allexcept": "single",
    }

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    label_counter = Counter()
    invalid_examples = []

    for item in raw_data:
        case = item.get("case", "").strip().lower()
        label = case_map.get(case)

        if label is None:
            invalid_examples.append(item)
            continue

        # Build JSON string as sentence
        example_json = {
            "question": item.get("question", ""),
            "options": item.get("options", ""),
            "answer": item.get("answer", ""),
        }
        sentence = json.dumps(example_json, ensure_ascii=False)
        processed.append({"sentence": sentence, "label": label})
        label_counter[label] += 1

    # Shuffle
    random.seed(seed)
    random.shuffle(processed)

    # Print label distribution
    total = sum(label_counter.values())
    print("🔢 Label distribution:")
    for label, count in label_counter.items():
        print(f"  {label}: {count} ({count / total:.2%})")
    print(f"  Total examples saved: {total}")

    # Report unmapped cases
    if invalid_examples:
        print(f"\n⚠️  Found {len(invalid_examples)} examples with unrecognized case values:")
        for i, ex in enumerate(invalid_examples[:10]):  # Print first 10 for brevity
            print(f"  [{i+1}] case: '{ex.get('case')}', question: {ex.get('question')}")
        if len(invalid_examples) > 10:
            print(f"  ...and {len(invalid_examples) - 10} more.")

    # Save to output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to: {output_path}")

if __name__ == "__main__":
    process_and_save_dataset()