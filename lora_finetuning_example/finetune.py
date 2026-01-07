import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# Create a small synthetic dataset
def create_synthetic_dataset():
    """Create a small sentiment classification dataset"""
    data = {
        'text': [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible film, waste of time and money. Very disappointing.",
            "Amazing performance by the actors. Highly recommend!",
            "Boring and predictable. I fell asleep halfway through.",
            "One of the best movies I've seen this year. Masterpiece!",
            "Poor script and terrible acting. Avoid at all costs.",
            "Brilliant cinematography and engaging story. Must watch!",
            "Completely awful. The worst movie I've ever seen.",
            "Excellent direction and beautiful visuals. Loved it!",
            "Dull and uninteresting. Regret watching this movie.",
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    }
    return Dataset.from_dict(data)

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    print("=" * 60)
    print("LoRA Fine-tuning Pipeline (Offline Mode)")
    print("=" * 60)

    # Configuration
    # NOTE: You need to download the model beforehand and specify the local path
    MODEL_PATH = "./bert-tiny"  # Change to your local model path
    OUTPUT_DIR = "./lora_finetuned_model"
    ADAPTER_DIR = "./lora_adapters"
    MAX_LENGTH = 128
    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 2e-4

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create synthetic dataset
    print("\n" + "=" * 60)
    print("Creating Synthetic Dataset")
    print("=" * 60)
    dataset = create_synthetic_dataset()

    # Split into train (8 samples) and test (2 samples)
    train_dataset = dataset.select(range(8))
    test_dataset = dataset.select(range(8, 10))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("\nSample data:")
    for i in range(2):
        print(f"  Text: {train_dataset[i]['text'][:60]}...")
        print(f"  Label: {'Positive' if train_dataset[i]['label'] == 1 else 'Negative'}\n")

    # Load tokenizer and model from local directory
    print("\n" + "=" * 60)
    print("Loading Model and Tokenizer")
    print("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=2,
            local_files_only=True
        )
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}")
        print(f"Error: {e}")
        return

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("\nTokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Configure LoRA
    print("\n" + "=" * 60)
    print("Configuring LoRA")
    print("=" * 60)
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=["query", "value"],  # Apply LoRA to attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=2,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none"  # Disable wandb/tensorboard
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("\n" + "=" * 60)
    print("Training Model with LoRA")
    print("=" * 60)
    train_result = trainer.train()

    print("\nTraining completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")

    # Save LoRA adapters only
    print("\n" + "=" * 60)
    print("Saving LoRA Adapters")
    print("=" * 60)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapters saved to: {ADAPTER_DIR}")

    # List saved files
    adapter_files = os.listdir(ADAPTER_DIR)
    print(f"Saved files: {adapter_files}")

    # Clean up memory
    del model
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load base model and apply saved adapters
    print("\n" + "=" * 60)
    print("Loading Base Model and Applying LoRA Adapters")
    print("=" * 60)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        local_files_only=True
    )
    model_with_adapters = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model_with_adapters.to(device)
    model_with_adapters.eval()

    print("Model loaded successfully with adapters!")

    # Inference on all test samples
    print("\n" + "=" * 60)
    print("Running Inference on Test Samples")
    print("=" * 60)

    for i, example in enumerate(test_dataset):
        text = example['text']
        true_label = example['label']

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model_with_adapters(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Get probabilities
        probs = torch.softmax(logits, dim=1)[0]
        confidence = probs[predicted_label].item()

        sentiment_map = {0: "Negative", 1: "Positive"}

        print(f"\n--- Test Example {i+1} ---")
        print(f"Text: {text}")
        print(f"True Label: {sentiment_map[true_label]}")
        print(f"Predicted: {sentiment_map[predicted_label]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")

    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Dataset: Synthetic sentiment classification (10 examples)")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Test accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"- Adapters saved to: {ADAPTER_DIR}")

if __name__ == "__main__":
    main()
