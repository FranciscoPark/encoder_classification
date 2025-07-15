from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Load tokenizer and model from your output directory
#model_dir = "microsoft/deberta-v3-small"  # or "./deberta-test-output/checkpoint-1000"
model_dir ="/home/s1/jypark/encoder_classification/deberta-test-output/checkpoint-5265"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

# Load and tokenize the SST-2 validation set
dataset = load_dataset("glue", "sst2")
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=256)

encoded = dataset.map(tokenize, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Set up eval arguments (NO deepspeed, NO mixed precision needed)
args = TrainingArguments(
    output_dir="./eval-logs",
    per_device_eval_batch_size=32,
    do_train=False,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics
)

# Run evaluation
metrics = trainer.evaluate(eval_dataset=encoded["validation"])
print(metrics)