import os
import json
import torch
import open_clip
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
STORAGE_PATH = "D:/_project/ImageMatch/storage"
FEEDBACK_FILE_PATH = os.path.join(STORAGE_PATH, "feedback.jsonl")
IMAGE_STORAGE_PATH = os.path.join(STORAGE_PATH, "images")
QUERIES_STORAGE_PATH = os.path.join(STORAGE_PATH, "queries")
FINETUNED_MODEL_PATH = os.path.join(STORAGE_PATH, "finetuned_model.pt")

BASE_MODEL_NAME = 'ViT-L-14'
PRETRAINED_DATASET = 'datacomp_xl_s13b_b90k'

# --- Hyperparameters ---
EPOCHS = 10 # More epochs for better learning
LEARNING_RATE = 1e-7 # A smaller learning rate is crucial for fine-tuning
BATCH_SIZE = 2 # Must be small due to memory usage of images
MARGIN = 0.2 # Margin for TripletLoss

# 1. Load Feedback and construct training triplets
def load_training_data():
    if not os.path.exists(FEEDBACK_FILE_PATH):
        print(f"Feedback file not found at {FEEDBACK_FILE_PATH}")
        return []

    feedback_triplets = []
    queries = defaultdict(lambda: {'Correct': set(), 'Incorrect': set()})

    with open(FEEDBACK_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                query_key = (data['query_image_filename'], data.get('query_text'))
                queries[query_key][data['judgment']].add(data['result_id'])
            except (json.JSONDecodeError, KeyError):
                print(f"Skipping malformed feedback line: {line.strip()}")
                continue

    # Create triplets of (anchor_query, positive_id, negative_id)
    for query_info, judgments in queries.items():
        for positive_id in judgments['Correct']:
            for negative_id in judgments['Incorrect']:
                feedback_triplets.append({
                    "query_image_filename": query_info[0],
                    "query_text": query_info[1],
                    "positive_id": positive_id,
                    "negative_id": negative_id
                })
    
    print(f"Generated {len(feedback_triplets)} training triplets from feedback.")
    return feedback_triplets

# 2. Create PyTorch Dataset
class TripletDataset(Dataset):
    def __init__(self, triplets, image_preprocess, text_tokenizer):
        self.triplets = triplets
        self.preprocess = image_preprocess
        self.tokenizer = text_tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        # Load images
        query_img_path = os.path.join(QUERIES_STORAGE_PATH, triplet["query_image_filename"])
        pos_img_path = os.path.join(IMAGE_STORAGE_PATH, triplet["positive_id"])
        neg_img_path = os.path.join(IMAGE_STORAGE_PATH, triplet["negative_id"])

        query_image = self.preprocess(Image.open(query_img_path).convert("RGB"))
        positive_image = self.preprocess(Image.open(pos_img_path).convert("RGB"))
        negative_image = self.preprocess(Image.open(neg_img_path).convert("RGB"))

        # Tokenize text
        query_text = self.tokenizer([triplet["query_text"] if triplet["query_text"] else ""])[0]

        return query_image, positive_image, negative_image, query_text

# 3. Main Training Function
def train():
    print("Starting REAL model fine-tuning process...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and preprocessors
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(BASE_MODEL_NAME, pretrained=PRETRAINED_DATASET)
    tokenizer = open_clip.get_tokenizer(BASE_MODEL_NAME)
    model.to(device)

    # Load data
    triplets = load_training_data()
    if not triplets:
        print("No training data available. Please provide more feedback.")
        return

    dataset = TripletDataset(triplets, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup loss and optimizer
    loss_fn = torch.nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train() # Set model to training mode
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for query_img, pos_img, neg_img, query_txt in progress_bar:
            query_img, pos_img, neg_img, query_txt = \
                query_img.to(device), pos_img.to(device), neg_img.to(device), query_txt.to(device)

            # Get embeddings from the model
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                query_img_vec = model.encode_image(query_img)
                pos_vec = model.encode_image(pos_img)
                neg_vec = model.encode_image(neg_img)

                # Combine query image and text vectors if text exists
                # Note: This is a simplified approach. A more advanced approach might use cross-attention.
                query_txt_vec = model.encode_text(query_txt)
                anchor_vec = query_img_vec + query_txt_vec
                anchor_vec = anchor_vec / anchor_vec.norm(dim=-1, keepdim=True)

                # Calculate loss
                loss = loss_fn(anchor_vec, pos_vec, neg_vec)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    print("\nTraining complete.")
    print(f"Saving fine-tuned model to {FINETUNED_MODEL_PATH}")
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
    print("\nFine-tuned model saved successfully.")
    print("You can now restart the main server to use the fine-tuned model.")

if __name__ == "__main__":
    train()