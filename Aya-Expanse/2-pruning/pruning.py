from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import sys
import torch


# Argument of the number of layers to remove
num_layers_remove = int(sys.argv[1])
if num_layers_remove > 16:
    print("num_layers_remove must be up to 16 layers")
print("Number of layers to remove:", num_layers_remove)

# The all_layers_to_remove list is a result of layer-importance-evaluation.py
all_layers_to_remove = [13, 12, 11, 16, 22, 5, 23, 9, 28, 10, 18, 26, 17, 7, 8, 15]
layers_to_remove = all_layers_to_remove[:num_layers_remove]
print("Layers to remove:", layers_to_remove)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Current device is:", device)

model_name = "CohereLabs/aya-expanse-8b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             ).to(device).eval()

# print(model)

model_parameters = sum(p.numel() for p in model.parameters())
print(f"Original model parameters: {model_parameters:,}")

# Check the currentl number of layers
current_num_layers = model.model.config.num_hidden_layers
print("Original model layers:", current_num_layers)

layers_to_keep = [l for l in range(current_num_layers) if l not in layers_to_remove]

lm_layers = model.model.layers

# Update the number of layers
model.model.layers = nn.ModuleList([lm_layers[n] for n in layers_to_keep])

# Update the number of layers in the config
model.config.num_hidden_layers = len(model.model.layers)

# Check the new number of parameters
model_parameters = sum(p.numel() for p in model.parameters())
print(f"New model parameters: {model_parameters:,}")

# Check the new number of layers
new_num_layers = model.model.config.num_hidden_layers
print("New model layers:" , new_num_layers)

user_id = "ymoslem/"  # Change to your HF user
new_model_name = f"{user_id}aya-expanse-8b-{new_num_layers}layers"
model.push_to_hub(new_model_name)
