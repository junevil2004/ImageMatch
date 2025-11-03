import os
import json
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from pydantic import BaseModel
import uuid

# --- Configuration ---
app = FastAPI()
STORAGE_PATH = "D:/_project/ImageMatch/storage"
IMAGE_STORAGE_PATH = os.path.join(STORAGE_PATH, "images")
VECTOR_STORAGE_PATH = os.path.join(STORAGE_PATH, "vectors.json")
FEEDBACK_FILE_PATH = os.path.join(STORAGE_PATH, "feedback.jsonl")
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files Mounting ---
app.mount("/storage", StaticFiles(directory=STORAGE_PATH), name="storage")

# --- Model Loading ---
print("Loading OpenCLIP model (ViT-L-14)...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Check for and load fine-tuned model if it exists
FINETUNED_MODEL_PATH = os.path.join(STORAGE_PATH, "finetuned_model.pt")
if os.path.exists(FINETUNED_MODEL_PATH):
    print(f"Found fine-tuned model at {FINETUNED_MODEL_PATH}. Loading weights.")
    try:
        model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
        print("Successfully loaded fine-tuned model weights.")
    except Exception as e:
        print(f"Error loading fine-tuned model weights: {e}. Using base model.")
else:
    print("No fine-tuned model found. Using base pre-trained model.")

print("Model loaded successfully.")

# --- Helper Functions ---
def load_vectors():
    if os.path.exists(VECTOR_STORAGE_PATH):
        with open(VECTOR_STORAGE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_vectors(vectors):
    with open(VECTOR_STORAGE_PATH, 'w') as f:
        json.dump(vectors, f, indent=4)

def generate_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().tolist()[0]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "ImageMatch API is running"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image
        file_extension = os.path.splitext(file.filename)[1]
        image_id = str(uuid.uuid4())
        image_filename = f"{image_id}{file_extension}"
        image_path = os.path.join(IMAGE_STORAGE_PATH, image_filename)

        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Generate and save the vector
        vector = generate_vector(image_path)
        vectors = load_vectors()
        vectors[image_filename] = vector
        save_vectors(vectors)

        return {"filename": image_filename, "vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/search/")
async def search_similar_images(
    file: UploadFile = File(...), 
    text_query: str = Form(None), 
    top_k: int = 5
):
    try:
        # Save the query image temporarily
        query_image_path = os.path.join(IMAGE_STORAGE_PATH, f"query_{file.filename}")
        with open(query_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Generate the vector for the query image
        image_vector = np.array(generate_vector(query_image_path))
        query_vector = image_vector

        # If text query is provided, combine vectors
        if text_query and text_query.strip():
            print(f"Received text query: {text_query}")
            text = tokenizer([text_query])
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            text_vector = text_features.cpu().numpy()[0]
            
            # Combine vectors and re-normalize
            combined_vector = image_vector + text_vector
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                query_vector = (combined_vector / norm).tolist()
            else:
                query_vector = combined_vector.tolist()

        os.remove(query_image_path)  # Clean up

        # Load stored vectors
        stored_vectors = load_vectors()
        if not stored_vectors:
            return {"results": []}

        filenames = list(stored_vectors.keys())
        vectors = np.array(list(stored_vectors.values()))

        # Calculate similarities
        similarities = cosine_similarity([query_vector], vectors)[0]

        # Get top_k results
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {"filename": filenames[i], "similarity": float(similarities[i])}
            for i in top_k_indices
        ]

        return {"results": results, "query_vector": query_vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/gallery/")
def get_gallery():
    try:
        vectors = load_vectors()
        return {"images": list(vectors.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Feedback Handling ---
class FeedbackItem(BaseModel):
    query_vector: List[float]
    result_filename: str
    judgment: str

@app.post("/feedback/")
async def receive_feedback(item: FeedbackItem):
    try:
        feedback_data = {
            "query_vector": item.query_vector,
            "result_filename": item.result_filename,
            "judgment": item.judgment
        }
        with open(FEEDBACK_FILE_PATH, "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")