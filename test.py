from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import wandb
import os

if __name__ == "__main__":
    #models = ["microsoft/deberta-v3-small","microsoft/deberta-v3-base","microsoft/deberta-v3-large"]
    model_name = "microsoft/deberta-v3-small"
    #model_name ="microsoft/deberta-v3-large"
    #model_name = "microsoft/deberta-v3-base"
    dataset = load_dataset("glue", "sst2")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(ex):
        return tokenizer(ex["sentence"], padding="max_length", truncation=True, max_length=1024)

    encoded_dataset = dataset.map(tokenize, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Init wandb
    wandb.init(project="pjm_encoder", name=model_name, mode="online" if int(os.environ.get("RANK", 0)) == 0 else "disabled")
    

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="./encoder/"+model_name,
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
    )

    trainer.train()
    wandb.finish()