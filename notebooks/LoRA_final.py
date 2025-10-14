import json
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import torch

# =====================================
#  WhatsApp Chat Preprocessing
# =====================================

# Load exported WhatsApp structured JSON
file_path = "../output/combined_text.json"
with open(file_path, "r", encoding="utf-8") as f:
    all_chats = json.load(f)

# Define your own WhatsApp name
my_name = "Anurada"  

# Prepare fine-tuning structured conversation format
fine_tuning_data = []

for chat in all_chats:
    messages = chat["messages"]
    conv = []
    for msg in messages:
        sender = msg["sender"]
        content = msg["message"].strip()
        if not content:
            continue  # skip empty messages

        role = "assistant" if sender == my_name else "user"
        conv.append({"role": role, "content": content})

    # Keep only conversations that have at least one assistant message
    if any(m["role"] == "assistant" for m in conv):
        fine_tuning_data.append(conv)

# Save structured fine-tuning data
output_path = "../output/fine_tuning/data/fine_tuning.json"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(fine_tuning_data, f, ensure_ascii=False, indent=4)

print(f"Saved fine-tuning data to {output_path}")

# =====================================
# 1. Load Fine-Tuning Data
# =====================================

with open(output_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Combine conversation turns (user + assistant)
combined_texts = []
for conv in data:
    for i in range(0, len(conv), 2):
        user_msg = conv[i].get("content", "")
        assistant_msg = conv[i + 1].get("content", "") if i + 1 < len(conv) else ""
        combined_texts.append(f"<s>{user_msg}</s><sep><s>{assistant_msg}</s>")

# =====================================
# 2. Load Tokenizer & Model
# =====================================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "sep_token": "<sep>"})

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =====================================
# 3. Apply LoRA
# =====================================

from transformer.lora import get_lora_model, print_trainable_parameters

lora_model = get_lora_model(
    model=model,
    lora_config={
        "rank": 4,
        "alpha": 8,
    },
    device=device
)
print_trainable_parameters(lora_model)

# =====================================
# 4. Tokenize Data
# =====================================

block_size = 128

# Tokenize all combined conversation texts
tokenized_data = tokenizer(
    combined_texts,
    truncation=True,
    padding="max_length",
    max_length=block_size,
    return_tensors="pt"
)

# Copy input_ids to labels
labels = tokenized_data["input_ids"].clone()

# Mask out the context part (everything before <sep>)
sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

for i, ids in enumerate(tokenized_data["input_ids"]):
    # Find where the <sep> token occurs
    sep_positions = (ids == sep_token_id).nonzero(as_tuple=True)[0]
    if len(sep_positions) > 0:
        sep_idx = sep_positions[0].item()
        # Mask all tokens before and including <sep>
        labels[i, :sep_idx + 1] = -100  # these tokens won't be used for loss

# Assign masked labels
tokenized_data["labels"] = labels

# =====================================
# 5. Build Hugging Face Dataset
# =====================================

dataset = Dataset.from_dict({
    "input_ids": tokenized_data["input_ids"],
    "attention_mask": tokenized_data["attention_mask"],
    "labels": tokenized_data["labels"]
})

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# =====================================
# 6. Training Arguments
# =====================================

training_args = TrainingArguments(
    output_dir="./results_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    report_to="none",
)

# =====================================
# 7. Trainer (LoRA model)
# =====================================

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# =====================================
# 8. Train LoRA model
# =====================================

trainer.train()

# =====================================
# 9. Visualize Loss
# =====================================

train_loss_steps = []
eval_loss_steps = []

for log in trainer.state.log_history:
    if "loss" in log:
        train_loss_steps.append((log["step"], log["loss"]))
    if "eval_loss" in log:
        eval_loss_steps.append((log["step"], log["eval_loss"]))

if train_loss_steps:
    steps, losses = zip(*train_loss_steps)
    plt.plot(steps, losses, label="Train Loss")
if eval_loss_steps:
    steps, losses = zip(*eval_loss_steps)
    plt.plot(steps, losses, label="Validation Loss")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (LoRA)")
plt.legend()
plt.grid()
plt.show()

# =====================================
# 10. Generate from LoRA model
# =====================================

user_message = "Salam labas"
input_tokens = tokenizer.encode(user_message, return_tensors="pt").to(device)

lora_model.eval()
with torch.no_grad():
    output = lora_model.generate(input_tokens=input_tokens, max_new_tokens=100)

print(f"User: {user_message}")
print(f"Assistant (LoRA): {tokenizer.decode(output[0].tolist())}")


