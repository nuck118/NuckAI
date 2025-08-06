import sys
import os

# Add the parent directory (NuckAI) to the system path
# This allows the script to find the 'training' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.nn import functional as F
from training.model import MultimodalNuckAi
from training.tokenizer import SimpleTokenizer
from torchvision import transforms
from PIL import Image

# --- Hyperparameters ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
BLOCK_SIZE = 256

# --- Image Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Multimodal Generation ---
@torch.no_grad()
def generate(model, tokenizer, start_text, image_path=None, num_tokens_to_generate=100, top_k=5):
    # Prepare text input
    encoded_text = tokenizer.encode(start_text)
    context = torch.tensor(encoded_text, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Prepare image input
    image_tensor = None
    if image_path:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return start_text

    # Generation loop
    for _ in range(num_tokens_to_generate):
        # We need to truncate the context if it gets too long
        context_condensed = context[:, -(BLOCK_SIZE):]

        # Get logits from the model, passing both image and text
        logits, _ = model(text_input=context_condensed, image_input=image_tensor)
        
        # Focus on the last token's logits
        logits = logits[:, -1, :]

        # Apply Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Get probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token to the context
        context = torch.cat((context, next_token), dim=1)

    # Decode and return the final text
    decoded_text = tokenizer.decode(context[0].tolist())
    return decoded_text

# --- Main Inference Function ---
def main():
    # Load tokenizer
    try:
        tokenizer = SimpleTokenizer.load('nuck_tokenizer.json')
    except FileNotFoundError:
        print("Tokenizer not found. Please run tokenizer.py first.")
        return

    vocab_size = len(tokenizer.vocab)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Initialize and load the multimodal model
    model = MultimodalNuckAi(vocab_size=vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load('nuck_multimodal_model.pth', map_location=DEVICE))
        print("Multimodal model loaded successfully.")
    except FileNotFoundError:
        print("Multimodal model not found. Please run training/train.py first.")
        return
    model.eval()

    # --- Test 1: Image + Text Prompt ---
    print("\n--- Test 1: Generate text from an image of a cat ---")
    image_path = os.path.join('data', 'multimodal_data', 'cat.jpg')
    start_text = "A picture of a"
    generated_text = generate(model, tokenizer, start_text, image_path=image_path, num_tokens_to_generate=30)
    print(f"Image: {image_path}\nPrompt: {start_text}\nOutput: {generated_text}\n")

    # --- Test 2: Image + Text Prompt ---
    print("--- Test 2: Generate text from an image of a car ---")
    image_path = os.path.join('data', 'multimodal_data', 'car.jpg')
    start_text = "The car is"
    generated_text = generate(model, tokenizer, start_text, image_path=image_path, num_tokens_to_generate=30)
    print(f"Image: {image_path}\nPrompt: {start_text}\nOutput: {generated_text}\n")

    # --- Test 3: Text-only prompt to show the model can still work without an image ---
    print("--- Test 3: Generate text without an image ---")
    start_text = "NuckAi is"
    generated_text = generate(model, tokenizer, start_text, image_path=None, num_tokens_to_generate=30)
    print(f"Prompt: {start_text}\nOutput: {generated_text}\n")


if __name__ == '__main__':
    main()