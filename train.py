import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# ===== 1. Set paths =====
model_path = "/data/zcx/modelscope_models/models/LLM-Research/Meta-Llama-3-8B-Instruct"
classification_data_path = "/data/zcx/llama3_finetune/data/classification_data.jsonl"
generation_data_path = "/data/zcx/llama3_finetune/data/generation_data.jsonl"

# ===== 2. Set environment variables =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===== 3. Load tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ===== 4. Load and split classification dataset =====
cls_dataset = load_dataset("json", data_files=classification_data_path)["train"]
cls_split = cls_dataset.train_test_split(test_size=0.2, seed=42)
cls_eval_split = cls_split["test"].train_test_split(test_size=0.5, seed=42)
cls_train, cls_eval = cls_split["train"], cls_eval_split["train"]

# ===== 5. Tokenization function for classification =====
def tokenize_cls(example):
    formatted_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(
        formatted_text, truncation=True, padding="max_length", max_length=512
    )
    tokenized["labels"] = example["label"]
    return tokenized

tokenized_cls_train = cls_train.map(tokenize_cls, remove_columns=cls_train.column_names)
tokenized_cls_eval = cls_eval.map(tokenize_cls, remove_columns=cls_eval.column_names)

# ===== 6. Quantization config =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# ===== 7. Load model and apply LoRA =====
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map=device_map,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)

# ===== 8. Training arguments for classification task =====
cls_args = TrainingArguments(
    output_dir="./llama3_lora_cls_output",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    logging_dir="./logs_cls"
)

# ===== 9. Trainer for classification task =====
trainer_cls = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=cls_args,
    train_dataset=tokenized_cls_train,
    eval_dataset=tokenized_cls_eval,
    data_collator=DefaultDataCollator(),
)

print("==== Stage 1: Starting classification training ====")
trainer_cls.train()
print("==== Stage 1 complete ====")

# ===== 10. Load and split generation dataset =====
gen_dataset = load_dataset("json", data_files=generation_data_path)["train"]
gen_split = gen_dataset.train_test_split(test_size=0.2, seed=42)
gen_eval_split = gen_split["test"].train_test_split(test_size=0.5, seed=42)
gen_train, gen_eval = gen_split["train"], gen_eval_split["train"]

# ===== 11. Tokenization function for generation task =====
def tokenize_gen(example):
    formatted_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

tokenized_gen_train = gen_train.map(tokenize_gen, remove_columns=gen_train.column_names)
tokenized_gen_eval = gen_eval.map(tokenize_gen, remove_columns=gen_eval.column_names)

# ===== 12. Training arguments for generation task =====
gen_args = TrainingArguments(
    output_dir="./llama3_lora_gen_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=200,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
    logging_dir="./logs_gen"
)

# ===== 13. Trainer for generation task =====
trainer_gen = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=gen_args,
    train_dataset=tokenized_gen_train,
    eval_dataset=tokenized_gen_eval,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("==== Stage 2: Starting generation training ====")
trainer_gen.train()
print("==== All training complete ====")
