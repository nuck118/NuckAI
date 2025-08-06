import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MultimodalNuckAi
from tokenizer import SimpleTokenizer
from PIL import Image
import os

# --- Hyperparameters ---
BLOCK_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
EVAL_INTERVAL = 100
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- Image Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Multimodal Dataset and DataLoader ---
class MultimodalDataset(Dataset):
    def __init__(self, captions_file, img_dir, tokenizer, block_size):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.preprocess = preprocess
        
        self.data = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    self.data.append({'image_file': parts[0], 'caption': parts[1]})

        self.encoded_captions = [self.tokenizer.encode(item['caption']) for item in self.data]
    
    def __len__(self):
        # We need to make sure our dataset is large enough for the dataloader
        # We will repeat our small dataset many times
        return len(self.data) * 1000

    def __getitem__(self, idx):
        # Use modulo to cycle through our small dataset
        data_idx = idx % len(self.data)
        item = self.data[data_idx]
        
        # Load and preprocess image
        img_path = os.path.join(self.img_dir, item['image_file'])
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        
        # Get encoded text
        encoded_text = self.encoded_captions[data_idx]
        
        # Pad the text to the correct block size
        padded_text = encoded_text + [0] * (self.block_size - len(encoded_text) - 1)
        if len(padded_text) >= self.block_size:
            padded_text = padded_text[:self.block_size-1]
        
        x = torch.tensor([0] + padded_text, dtype=torch.long)
        y = torch.tensor(padded_text + [0], dtype=torch.long)
        
        return image, x, y

# --- Main Training Loop ---
def train():
    # Load tokenizer
    try:
        tokenizer = SimpleTokenizer.load('nuck_tokenizer.json')
    except FileNotFoundError:
        print("Tokenizer not found. Please run tokenizer.py first.")
        return

    vocab_size = len(tokenizer.vocab)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Create dataset
    # Use an absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Construct paths from the project root
    captions_path = os.path.join(project_root, 'data', 'multimodal_data', 'captions.txt')
    img_dir = os.path.join(project_root, 'data', 'multimodal_data')
    dataset = MultimodalDataset(captions_path, img_dir, tokenizer, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = MultimodalNuckAi(vocab_size=vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop using epochs
    print(f"Starting training on device: {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        for i, (images, xb, yb) in enumerate(dataloader):
            images, xb, yb = images.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
            
            # Pass image and text to the model
            logits, loss = model(text_input=xb, image_input=images, targets=yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if i % EVAL_INTERVAL == 0:
                print(f"  Iteration {i}: Loss = {loss.item():.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'nuck_multimodal_model.pth')
    print("Training complete. Multimodal model saved to nuck_multimodal_model.pth")

if __name__ == '__main__':
    train()