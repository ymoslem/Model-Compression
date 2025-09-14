# Requirements: pip3 install vllm datasets sacrebleu unbabel-comet
# Example usage: python3 inference.py deu 24

from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
import gc
import os
import polars as pl
import sys
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "false"

tgt_lang = sys.argv[1]  # "arz" or "deu"
model_layers = sys.argv[2]  # 40, 32, 24, 20, or 16


# Data
if tgt_lang == "deu":
    src_lang = "ces"
    full_src_lang = "Czech"
    full_tgt_lang = "German"

    dataset_name = "ymoslem/news-commentary-cs-de"

    dataset = load_dataset(dataset_name,
                           split="train",
                          )

    dataset = dataset.shuffle(seed=0)

    # Split dataset into train and test
    dataset = dataset.train_test_split(test_size=500, seed=0)

    dataset = dataset["test"]

elif tgt_lang == "arz":
    src_lang = "eng"
    full_src_lang = "English"
    full_tgt_lang = "Egyptian Arabic"

    dataset_name = "ymoslem/news-commentary-eng-arz"

    dataset = load_dataset(dataset_name,
                           split="test",
                          )
    dataset = dataset.rename_columns({"english": "source", "egyptian_arabic": "target"})

else:
    raise ValueError("Invalid tgt_lang or domain.")

print(dataset)


source_sentences = dataset["source"]
prompt = f"Translate the following text from {full_src_lang} to {full_tgt_lang}:"
prompts = [prompt + "\n" + sent + "\n" for sent in source_sentences]
print(prompts[0])


references = dataset["target"]
references[0]


def define_max_len(sentences):
    max_len, longest_idx = max([(len(sent.split()), idx)
                                for idx, sent in enumerate(sentences)])
    max_len = max_len * 2
    return max_len, longest_idx

max_len, longest_idx = define_max_len(source_sentences)

print(max_len)


# Model
if model_layers == "40":
    model_name = "CohereLabs/aya-expanse-32b"
elif model_layers == "32":
    model_name = "CohereLabs/aya-expanse-8b"
elif model_layers in ["16", "20", "24"]:
    model_name = f"ymoslem/wmt25-{src_lang}-{tgt_lang}-{model_layers}layers-2e-5lr-news-commentary"

    # or for KD models (deu only):
    # model_name = f"ymoslem/wmt25-{src_lang}-{tgt_lang}-{model_layers}layers-2e-05-100k-news-commentary-sentences-kd"
else:
    print("Inaccurate configuration. Please revise the model_layers")


num_gpus = torch.cuda.device_count()
awq = True if "-awq" in model_name.lower() else False  # verify based on your model
max_model_len = 4096


print(f"Model name: {model_name}")
print(f"Number of GPUs: {num_gpus}")
print(f"Max length: {max_len}")
print(f"AWQ: {awq}\n")


if awq:
    llm = LLM(model=model_name,
              tensor_parallel_size=num_gpus,
              quantization="awq_marlin",
              max_model_len=max_model_len,
              trust_remote_code=True,
              # download_dir=cache_dir,
             )
else:
    llm = LLM(model=model_name,
              dtype=torch.bfloat16,
              tensor_parallel_size=num_gpus,
              max_model_len=max_model_len,
              trust_remote_code=True,
              # download_dir=cache_dir,
              )


# Translation
print(f"Translating {len(prompts)} prompts...")

# Set up sampling parameters
sampling_params = SamplingParams(
                                temperature=0.0,  # Greedy decoding
                                max_tokens=max_len,
                                stop_token_ids=[llm.get_tokenizer().eos_token_id],
                                )

# Format all prompts for chat (if using instruct model)
formatted_prompts = []
for prompt in tqdm(prompts, desc="Formatting prompts"):
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    formatted_prompts.append(formatted_prompt)

# Generate all responses at once (vLLM handles batching internally)
print("\nGenerating responses...")
batch_outputs = llm.generate(formatted_prompts, sampling_params)

# Extract the generated text
translations = []
for output in batch_outputs:
    generated_text = output.outputs[0].text.strip()
    translations.append(generated_text)

print(f"Generated {len(translations)} responses\n")


translations[0]


# Optional: Save the translations to a file
with open(f"translations_{model_layers}layers.txt", "w") as output:
    for sentence in translations:
        output.write(sentence.strip() + "\n")


# Release memory
def release_memory(model):

    del model

    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

release_memory(llm)


# Evaluation

all_scores = []
chrf = CHRF(word_order=2)
chrf_score = round(chrf.corpus_score(translations, [references]).score, 2)
all_scores.append(chrf_score)


# Download and load a COMET model
comet_model_names = ["Unbabel/wmt20-comet-da", "Unbabel/wmt22-comet-da"]

for comet_model_name in comet_model_names:

    model_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(model_path).to("cuda")

    assert comet_model.device.type == "cuda"

    # Prepare the data
    data = []
    for src, mt, ref in zip(source_sentences, translations, references):
        data.append({
            "src": src,
            "mt": mt,
            "ref": ref
        })

    # Calculate COMET scores
    model_output = comet_model.predict(data, batch_size=8, gpus=1)
    comet_scores = model_output.scores
    comet_corpus_score = round(model_output.system_score * 100, 2)
    all_scores.append(comet_corpus_score)

    release_memory(comet_model)


# Print results
print(f"\nModel name: {model_name}")
df = pl.DataFrame([all_scores],
                  schema=["chrF++", "COMET20", "COMET22"],
                  orient="row",
                 )
print(df)
