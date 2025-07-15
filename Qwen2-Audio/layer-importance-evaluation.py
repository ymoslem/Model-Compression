# Iterative pruning (decoder layers only)

import gc
import logging
import os
import pandas as pd
import polars as pl
import torch
import sacrebleu
from datasets import load_dataset, Audio
from comet import download_model, load_from_checkpoint
from copy import deepcopy
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
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


data_cache_dir = "/workspace/data/"
model_cache_dir = "/workspace/model/"


# tgt_lang_code = "de"
tgt_lang_code = "zh"


# Load the model

# model_name = "Qwen/Qwen2-Audio-7B-Instruct"

if tgt_lang_code == "de":
    model_name = "ymoslem/qwen-audio-en-de-4bs-1acc-1e-05lr-cosine-0.0warmup-0.001wd-3epochs-full-new"
elif tgt_lang_code == "zh":
    model_name = "ymoslem/qwen-audio-en-zh-4bs-1e-05lr-cosine-0.001wd-3epochs-acl6060-full-new"
else:
    print("Please select either 'de' or 'zh'.")

pretrained_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name,
                                                           torch_dtype=torch.bfloat16,
                                                           #cache_dir=model_cache_dir,
                                                          ).to("cuda").eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("Model loaded:", model_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

model_parameters = count_parameters(pretrained_model)
print(f"Model parameters: {model_parameters:,}")


print("Current values:",
      pretrained_model.config.audio_config.encoder_layers,
      pretrained_model.config.audio_config.num_hidden_layers,
      pretrained_model.language_model.model.config.num_hidden_layers
     )


# Load the dataset

acl6060_all = load_dataset("ymoslem/ACL-6060",
                           split="dev+eval",
                           cache_dir=data_cache_dir
                          )
acl6060_all = acl6060_all.cast_column("audio", Audio(sampling_rate=16000))

acl6060 = acl6060_all.train_test_split(test_size=100, seed=0)

print(acl6060)



def translate(audio_array, audio_path, sr, language, shot=0, model_type="instruct"):
    
    if model_type == "base":
        text = f"<|audio_bos|><|AUDIO|><|audio_eos|>Translate the English speech into {language}:"

    elif model_type == "instruct":
        if shot == 1:
            conversation = [
                {"role": "system", "content": f"You are a professional translator."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"As knowledge base we use Wikipedia. \
                    Translate the English speech into {language}:"},
                ]},
                {"role": "assistant", "content": f"Als Wissensbasis verwenden wir Wikipedia."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path},  # just for formatting
                    {"type": "text", "text": f"Translate the English speech into {language}:"},
                ]},
            ]

        else:
            conversation = [
                {"role": "system", "content": f"You are a professional translator."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": f"Translate the English speech into {language}:"},
                ]},
            ]


        text = processor.apply_chat_template(conversation,
                                             add_generation_prompt=True,
                                             tokenize=False,
                                            )


    inputs = processor(text=text,
                       audio=audio_array,
                       sampling_rate=sr,
                       return_tensors="pt").to("cuda")

    
    max_length = 1024
    generate_ids = model.generate(**inputs,
                                  max_length=max_length,
                                  do_sample=False,
                                  repetition_penalty=1.0,
                                  pad_token_id=processor.tokenizer.eos_token_id,
                                 )
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)[0]

    return response.strip()


shot = 0
model_type = "instruct"

if tgt_lang_code == "de":
    language = "German"
elif tgt_lang_code == "zh":
    language = "Chinese"
elif tgt_lang_code == "ar":
    language = "Arabic"
else:
    raise ValueError(f"Unsupported target language code: {tgt_lang_code}")

print(f"{model_name=}\n{model_type=}\n{language=}")


references = acl6060["test"][f"text_{tgt_lang_code}"]
source_sentences = acl6060["test"]["text_en"]

print(references[0])
print(source_sentences[0])


# Load COMET

model_path = download_model("wmt20-comet-da")
comet_model = load_from_checkpoint(model_path)

# Create the json path to save jsonl files
os.makedirs("json", exist_ok=True) 

# Downscaling

all_dfs = []

total_remove = 8

layers_to_remove = []
ignore = set([0, 31] + layers_to_remove)
full = set(range(32))


for i in tqdm(range(total_remove), desc="Downscaling", total=total_remove):

    print(f"\n---Run {len(layers_to_remove)} - Layers removed: {layers_to_remove}---")

    main_df = pl.DataFrame({
                            "Layer": [],
                            "BLEU": [],
                            "ChrF++": [],
                            "COMET": [],
                            "ChrF": [],
                            }).cast({
                                "Layer": pl.Int64,
                                "BLEU": pl.Float64,
                                "ChrF++": pl.Float64,
                                "COMET": pl.Float64,
                                "ChrF": pl.Float64,
                            })

    best_score = float('-inf')
    best_layer = None

    required = sorted(list(full - ignore))

    for n in tqdm(required, total=len(required), position=1, desc="Layers eval"):

        print("\nPruning layer", n, "\n")

        layers_to_keep = list(full - set(layers_to_remove + [n]))

        model = None
        gc.collect()
        with torch.no_grad():
            empty_cache()

        model = deepcopy(pretrained_model)

        lm_layers = model.language_model.model.layers

        model.language_model.model.layers = nn.ModuleList([lm_layers[n] for n in layers_to_keep])

        # Ensure the config reflects the actual number of layers
        model.config.text_config.num_hidden_layers = len(model.language_model.model.layers)
        model.language_model.model.config.num_hidden_layers = len(model.language_model.model.layers)


        # Check the new number of parameters
        model_parameters = count_parameters(model)
        print(f"Model parameters: {model_parameters:,}")

        # Check the new number of layers
        print("New values:" ,
              model.config.audio_config.encoder_layers,
              model.config.audio_config.num_hidden_layers,
              model.language_model.model.config.num_hidden_layers,
             )


        # Translation

        # print("Original config:", model.generation_config)
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        # print("Modified config:", model.generation_config)

        translations = []

        for segment in tqdm(acl6060["test"], desc="Translating"):
            audios = [segment["audio"]["array"]]
            audio_path = segment["audio"]["path"]
            sr = segment["audio"]["sampling_rate"]

            translation = translate(audios, audio_path, sr, language, shot, model_type)
            translations.append(translation)


        # Evaluation

        bleu_tokenizer = "zh" if tgt_lang_code == "zh" else None

        # Calculate BLEU
        bleu_score = sacrebleu.corpus_bleu(translations, [references], tokenize=bleu_tokenizer)
        bleu_score = round(bleu_score.score, 2)

        # Calculate ChrF++
        chrf_pp_score = sacrebleu.corpus_chrf(translations, [references], word_order=2)
        chrf_pp_score = round(chrf_pp_score.score, 2)

        chrf_score = sacrebleu.corpus_chrf(translations, [references])
        chrf_score = round(chrf_score.score, 2)

        # Calculate COMET
        df = pd.DataFrame({"src":source_sentences, "mt":translations, "ref":references})
        data = df.to_dict('records')

        seg_scores, sys_score = comet_model.predict(data, batch_size=128, gpus=1).values()
        comet_score = round(sys_score*100, 2)

        if tgt_lang_code == "de":
            metric_score = chrf_pp_score  # chrF++
            print(f"\n{metric_score=}")
        elif tgt_lang_code == "zh":
            metric_score = chrf_score  # chrF
            print(f"\n{metric_score=}")
        else:
            print("Please select either 'de' or 'zh'.")

        if metric_score > best_score:
            best_score = metric_score
            best_layer = n

        print("Best layer to remove so far:", best_layer, "Score:", best_score)

        df = pl.DataFrame({"Layer": n,
                           "BLEU": bleu_score,
                           "ChrF++": chrf_pp_score,
                           "COMET": comet_score,
                           "ChrF": chrf_score,
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
