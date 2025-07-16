import os
import json
import wandb
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_custom_dataset(data_path, tokenizer):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label2id = {"special": 0, "log": 1, "single": 2}

    for item in data:
        item["label"] = label2id[item["label"]]

    dataset = Dataset.from_list(data)

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    model_name = "microsoft/deberta-v3-small"
    #model_name ="microsoft/deberta-v3-large"
    #model_name = "microsoft/deberta-v3-base"
    data_path = "/home/s1/jypark/encoder_classification/trainset/processed.json"
    output_dir = f"./encoder/{model_name}"
    validation_path = "/home/s1/jypark/encoder_classification/validationset/processed.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #full_dataset = load_custom_dataset(data_path, tokenizer)

    # Split into 90% train, 10% test
    #dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = load_custom_dataset(data_path, tokenizer)
    eval_dataset = load_custom_dataset(validation_path,tokenizer)

    # WandB
    wandb.init(project="pjm_encoder", name=model_name, mode="online" if int(os.environ.get("RANK", 0)) == 0 else "disabled")

    # Model (3 labels: special, log, single)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_dir="./logs",
        bf16=True,
        report_to="wandb",
        save_strategy="epoch",
        deepspeed="/home/s1/jypark/encoder_classification/ds_z0_config.json"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final evaluation on best model
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("\n🔍 Final evaluation on test set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.finish()