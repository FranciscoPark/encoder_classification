# train.py
import os
import argparse
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from dataset import load_and_tokenize_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-small")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--wandb_project", default="deberta-finetune")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    # Load dataset
    dataset = load_and_tokenize_dataset(args.model_name, args.data_path, num_labels=args.num_labels)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="wandb",
        deepspeed=args.deepspeed,
        fp16=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()