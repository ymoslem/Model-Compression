from datasets import load_dataset
from huggingface_hub import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
import json
import torch


# Language codes
src_lang = "cs"     # can be one of {'en', 'cs', 'ja'}
tgt_lang = "de_DE"  # can be one of {'ar_EG', 'de_DE', 'zh_CN'}

full_src_lang = "Czech"
full_tgt_lang = "German"

cache_dir = "./cache"

# Dataset ("ymoslem/news-commentary-cs-de") -- sentence-level data
dataset_name = f"ymoslem/news-commentary-{src_lang}-{tgt_lang.split('_')[0]}"

dataset = load_dataset(dataset_name,
                       split="train",
                       cache_dir=cache_dir,
                      )

print(f"Loaded the dataset: {dataset_name}")

prompt = f"Translate the following text from {full_src_lang} to {full_tgt_lang}:"
train_prompts = [prompt] * dataset.num_rows

dataset = dataset.add_column("prompt", train_prompts)
dataset = dataset.shuffle(seed=0)

# Split dataset into train and test
dataset = dataset.train_test_split(test_size=500, seed=0)

# Take a portion of the train dataset
dataset["train"] = dataset["train"].select(range(100000))
train_num_rows = dataset["train"].num_rows
eval_num_rows = dataset["test"].num_rows
print(f"{train_num_rows=}, {eval_num_rows=}")
print(dataset)


# Model
num_layers = 24  # 24, 20, or 16
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_name = "CohereLabs/aya-expanse-8b"
model_name = f"ymoslem/aya-expanse-8b-{num_layers}layers-{src_lang}-{tgt_lang.split('_')[0]}"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             use_cache=False,
                                             cache_dir=cache_dir,
                                             device_map="auto",
                                             )
assert model.device.type == "cuda"
print(f"\nModel loaded on {model.device.type}: {model_name}")

# Formatting function for batched processing of the data
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i] + "\n" + examples["source"][i] + "\n"
        target = examples["target"][i]
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{target}"
        output_texts.append(text)
    return {"text": output_texts}

# Apply formatting to the datasets
train_dataset_formatted = dataset["train"].map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset["train"].column_names
)

eval_dataset_formatted = dataset["test"].map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset["test"].column_names
)

print("\nTraining data example:", train_dataset_formatted[0])

# Training configuration
num_epochs = 1
eval_steps= 100
save_steps = 500
train_batch_size = 8
eval_batch_size = 8
grad_acc_steps = 1
use_grad_checkpointing = True
learning_rate = 2e-5 # or 1e-5

# Output repository
hf_id = "ymoslem/"  # change to your user name
details = f"{int(train_num_rows/1000)}k-news-commentary-sentences"
output_dir = f"{hf_id}wmt25-{src_lang}-{tgt_lang[:2]}-{num_layers}layers-{learning_rate}-{details}"


training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    gradient_checkpointing=use_grad_checkpointing,
    optim="paged_adamw_32bit",
    learning_rate=learning_rate,
    lr_scheduler_type="cosine", # or "constant",
    # weight_decay=0.001,  # use for full training of the teacher
    fp16=False,
    bf16=True,
    # warmup_ratio=0.05,
    group_by_length=True,

    # Evaluation
    eval_strategy="steps",
    eval_steps=eval_steps,
    per_device_eval_batch_size=eval_batch_size,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",

    # Saving
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,
    logging_steps=10,

    # push to hub parameters
    push_to_hub=True,
    hub_private_repo=True,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_formatted,
    eval_dataset=eval_dataset_formatted,
    processing_class=tokenizer,
    args=training_arguments,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]  # + load_best_model_at_end=True
)

print(f"Training Arguments {json.dumps(training_arguments.to_dict(), indent=2)}")

# Start training
print("\nTraining...")
trainer.train()

print("\nTraining is complete.")
