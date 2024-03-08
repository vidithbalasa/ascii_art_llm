# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install -U accelerate bitsandbytes datasets peft transformers

# + active=""
# 1. Load Open Assistant dataset

# +
from dataset import load_ascii_dataset

dataset = load_ascii_dataset()
# -

dataset

print(dataset[1]["text"])

# + active=""
# 2. Load and prepare model and tokenizer

# +
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

modelpath="mistralai/Mistral-7B-v0.1"

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)   

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token 
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id
# -

# Add LoRA adapters to model
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
    lora_dropout=0.1, 
    bias="none", 
    modules_to_save = ["lm_head", "embed_tokens"],		# needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

model

# + active=""
# 3. Prepare data for Training

# +
import os 

def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)
# -

dataset_tokenized


# define collate function - transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }
    return batch


# + active=""
#  Training Hyperparameters

# +
bs=5
ga_steps=1  # gradient acc. steps
epochs=100
steps_per_epoch=len(dataset_tokenized)//(bs*ga_steps)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,		# eval and save once per epoch  	
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.0002,
    group_by_length=True,
    fp16=True,
    ddp_find_unused_parameters=False,
)

# +
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=dataset_tokenized,
    eval_dataset=dataset_tokenized,
    args=args,
)

trainer.train()
# -

# Merge models

# +
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_path="mistralai/Mistral-7B-v0.1"  # input: base model
adapter_path="out/checkpoint-522"       # input: adapters
save_to="models/Mistral-7B-ascii-art-qlora"       # out: merged model ready for inference

base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_path)

# Add/set tokens same tokens to base model before merging, like we did before starting training https://github.com/geronimi73/qlora-minimal/blob/main/qlora.py#L27  
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.eos_token_id = tokenizer.eos_token_id

# Load LoRA and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
