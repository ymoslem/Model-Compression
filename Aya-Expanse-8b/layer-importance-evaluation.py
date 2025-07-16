import gc
import logging
import os
import polars as pl
import torch
import sacrebleu
from datasets import load_dataset
from statistics import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as tf_logging
from torch import nn
from torch.cuda import empty_cache
from tqdm.auto import tqdm


# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["POSSIBLE_USER_WARNINGS"] = "off"

tf_logging.set_verbosity_error()

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch.lightning").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load the tokenizer
model_name = "CohereLabs/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.bfloat16,
                                                ).to(device).eval()

    assert model.device.type == "cuda"

    print("\nOriginal model loaded:\n", model_name)

    model_parameters = count_parameters(model)
    print(f"\nModel parameters\n: {model_parameters:,}")

    print("\nOriginal number of layers\n:",
        model.model.config.num_hidden_layers
        )

    return model


# Load the dataset
dataset_name = "ymoslem/wmt2025-test-mt"
dataset = load_dataset(dataset_name,
                       split="test"
                      )

src_lang = "cs"
tgt_lang = "de_DE"  # can be one of {'ar_EG', 'de_DE', 'zh_CN'}

num_rows = 200  # Change if needed

dataset = dataset.filter(lambda x: x["src_lang"] == src_lang and x["tgt_lang"] == tgt_lang)

dataset = dataset.shuffle(seed=0)
dataset = dataset.select(range(num_rows))

print("Dataset loaded:", dataset_name)

prompt_instructions = dataset["prompt_instruction_pr"]
source_sentences = dataset["source_paragraphs"]
references = dataset["GPT4.1"]
tgt_langs = dataset["tgt_lang"]
src_langs = dataset["src_lang"]

assert len(prompt_instructions) == len(source_sentences) == len(src_langs) == len(tgt_langs)

print("Dataset rows:", dataset.num_rows)

prompts = [f"{prompt}\n\n{source}" for prompt, source in zip(prompt_instructions, source_sentences) ]

print(f"Prompts: {len(prompts)}")
print(f"Example prompt:\n{prompts[10]}")

def define_max_len(tgt_lang):
    lens = [len(sent.split())
            for sent, lang in zip(source_sentences, tgt_langs)
            if lang == tgt_lang]
    max_len = int(mean(lens) * 3)
    return max_len

max_len = define_max_len(tgt_lang)

print(max_len)


# Translate in batches

def translate(prompts, model):

    batch_size = num_rows  # If memory does not allow, it should be smaller.
    print("Batch Size:" batch_size)

    translations = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        # Format all messages in the batch
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

        # Tokenize the entire batch and get attention mask
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True  # This returns both input_ids and attention_mask
        )

        input_ids = batch_inputs['input_ids'].to(device)
        attention_mask = batch_inputs['attention_mask'].to(device)

        # Store original lengths for each sequence in the batch
        original_length = input_ids.shape[1]  # All sequences have same length due to padding

        # Generate for the entire batch with attention mask
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_new_tokens=max_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode batch results
        for j, tokens in enumerate(gen_tokens):
            # Get the length of the original input for this specific sequence
            new_tokens = tokens[original_length:]
            translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            translations.append(translation)

    return translations


# Downscaling

all_dfs = []

total_remove = 16  # Change if needed
print("Total layers to remove:", total_remove)

layers_to_remove = []
ignore = set([0, 31] + layers_to_remove)
full = set(range(32))

for i in tqdm(range(total_remove), desc="Downscaling", total=total_remove):

    print(f"\n---Run {len(layers_to_remove)} - Layers removed: {layers_to_remove}---")

    main_df = pl.DataFrame({
                            "Layer": [],
                            "ChrF": [],
                            }).cast({
                                "Layer": pl.Int64,
                                "ChrF": pl.Float64,
                            })

    best_score = float('-inf')
    best_layer = None

    required = sorted(list(full - ignore))

    for n in tqdm(required, total=len(required), position=1, desc="Layers eval"):

        print("\nPruning layer", n, "\n")

        # Clear any existing model
        if 'model' in locals():
            del model

        model = None
        gc.collect()
        with torch.no_grad():
            empty_cache()
        
        # Load the model
        model = load_model(model_name)

        layers_to_keep = list(full - set(layers_to_remove + [n]))

        lm_layers = model.model.layers

        # Update the number of layers
        model.model.layers = nn.ModuleList([lm_layers[n] for n in layers_to_keep])

        # Ensure the config reflects the actual number of layers
        model.config.num_hidden_layers = len(model.model.layers)

        # Check the new number of parameters
        model_parameters = count_parameters(model)
        print(f"New model parameters: {model_parameters:,}")

        # Check the new number of layers
        print("New number of layers:",
              model.model.config.num_hidden_layers,
             )


        # Translation

        translations = translate(prompts, model)


        # Evaluation

        print("Evaluation:")

        if tgt_lang == "de_DE" or tgt_lang == "ar_EG":  # use chrF++
            metric_score = sacrebleu.corpus_chrf(translations, [references], word_order=2)
            metric_score = round(metric_score.score, 2)
            print(f"\n{metric_score=}\n")
        elif tgt_lang == "zh_CH":  # use chrF
            metric_score = sacrebleu.corpus_chrf(translations, [references])
            metric_score = round(metric_score.score, 2)
            print(f"\n{metric_score=}\n")
        else:
            print("Please select either 'de_DE', 'ar_EG', or 'zh_CH'.")

        if metric_score > best_score:
            best_score = metric_score
            best_layer = n

        print("Best layer to remove so far:", best_layer, "Score:", best_score)

        df = pl.DataFrame({"Layer": n,
                           "ChrF": metric_score,
                          }
                         )

        main_df = main_df.vstack(df)

        print(df)

    layers_to_remove.append(best_layer)
    ignore.add(best_layer)
    print("\nLayers to remove:", layers_to_remove)


    all_dfs.append(main_df)
    main_df.write_ndjson(f"json/df_{i}.ndjson")

    with pl.Config(tbl_rows=32):
        print(main_df)

with pl.Config(tbl_rows=32):
    for layer_df in all_dfs:
        print(layer_df)

print("End of evaluation. Layers to remove:", layers_to_remove)
