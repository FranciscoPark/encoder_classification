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
    return dataset, data  # return raw data too for logging predictions

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
    is_main_process = int(os.environ.get("RANK", 0)) == 0

    #model_name = "microsoft/deberta-v3-small"
    #model_name ="microsoft/deberta-v3-large"
    model_name = "microsoft/deberta-v3-base"
    data_path = "/home/s1/jypark/encoder_classification/trainset/processed.json"
    validation_path = "/home/s1/jypark/encoder_classification/validationset/processed.json"
    output_dir = f"./encoder/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, _ = load_custom_dataset(data_path, tokenizer)
    eval_dataset, raw_eval_data = load_custom_dataset(validation_path, tokenizer)

    if is_main_process:
        wandb.init(project="pjm_encoder", name=model_name, mode="online")
    else:
        wandb.init(mode="disabled")

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

   
    id2label = {0: "special", 1: "log", 2: "single"}

    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = logits.argmax(-1)

    # Confusion Matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=labels,
            preds=preds,
            class_names=["special", "log", "single"]
        )
    })

    # Full predictions table
    table = wandb.Table(columns=["sentence", "gold", "predicted", "correct"])
    for i, (pred, label) in enumerate(zip(preds, labels)):
        sent = raw_eval_data[i]["sentence"]
        correct = pred == label
        table.add_data(sent, id2label[label], id2label[pred], correct)
    wandb.log({"eval_predictions": table})

    # Wrong predictions table
    wrong_table = wandb.Table(columns=["sentence", "gold", "predicted"])
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred != label:
            sent = raw_eval_data[i]["sentence"]
            wrong_table.add_data(sent, id2label[label], id2label[pred])
    wandb.log({"wrong_predictions": wrong_table})

    acc = accuracy_score(labels, preds)
    prec, rec, f1 = precision_recall_fscore_support(labels, preds, average="macro")[:3]

    print("\n🔍 Final Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    wandb.finish()