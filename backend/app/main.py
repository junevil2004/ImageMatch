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

# --- Helper Functions (Refactored for new data structure) ---
def load_vectors():
    """Loads a list of vector objects from the JSON file."""
    if os.path.exists(VECTOR_STORAGE_PATH) and os.path.getsize(VECTOR_STORAGE_PATH) > 0:
        try:
            with open(VECTOR_STORAGE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty list if file is corrupted
    return [] # Return empty list if file doesn't exist or is empty

def save_vectors(vector_list):
    """Saves a list of vector objects to the JSON file."""
    with open(VECTOR_STORAGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(vector_list, f, indent=4)

def generate_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().tolist()[0]

# --- API Endpoints (Refactored) ---
@app.get("/")
def read_root():
    return {"message": "ImageMatch API is running"}

@app.post("/uploads/")
async def upload_images(files: List[UploadFile] = File(...)):
    vector_data = load_vectors()
    newly_uploaded_files = []
    try:
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            image_id = f"{str(uuid.uuid4())}{file_extension}"
            image_path = os.path.join(IMAGE_STORAGE_PATH, image_id)

            with open(image_path, "wb") as buffer:
                buffer.write(await file.read())

            vector = generate_vector(image_path)
            
            vector_data.append({
                "id": image_id,
                "original_filename": file.filename,
                "vector": vector
            })
            newly_uploaded_files.append(file.filename)

        save_vectors(vector_data)
        return {"filenames": newly_uploaded_files, "message": f"{len(newly_uploaded_files)} files uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/search/")
async def search_similar_images(
    file: UploadFile = File(...), 
    text_query: str = Form(None), 
    top_k: int = 5
):
    vector_data = load_vectors()
    if not vector_data:
        return {"results": [], "query_vector": []}

    try:
        query_image_path = os.path.join(IMAGE_STORAGE_PATH, f"query_{file.filename}")
        with open(query_image_path, "wb") as buffer:
            buffer.write(await file.read())

        image_vector = np.array(generate_vector(query_image_path))
        query_vector = image_vector

        if text_query and text_query.strip():
            print(f"Received text query: {text_query}")
            text = tokenizer([text_query])
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_vector = text_features.cpu().numpy()[0]
            combined_vector = image_vector + text_vector
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                query_vector = (combined_vector / norm)
        
        os.remove(query_image_path)

        # Prepare data for cosine similarity
        gallery_vectors = np.array([item['vector'] for item in vector_data])
        
        similarities = cosine_similarity([query_vector], gallery_vectors)[0]
        
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for i in top_k_indices:
            item = vector_data[i]
            results.append({
                "id": item['id'],
                "original_filename": item['original_filename'],
                "similarity": float(similarities[i])
            })

        return {"results": results, "query_vector": query_vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/gallery/")
def get_gallery():
    try:
        vector_data = load_vectors()
        # Return a simplified list for the gallery view
        gallery_items = [{"id": item["id"], "original_filename": item["original_filename"]} for item in vector_data]
        return {"images": gallery_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Feedback Handling ---
class FeedbackItem(BaseModel):
    query_vector: List[float]
    result_filename: str # This is the UUID filename (the 'id')
    judgment: str

@app.post("/feedback/")
async def receive_feedback(item: FeedbackItem):
    try:
        feedback_data = {
            "query_vector": item.query_vector,
            "result_id": item.result_filename, # Renaming for clarity
            "judgment": item.judgment
        }
        with open(FEEDBACK_FILE_PATH, "a", encoding='utf-8') as f:
            f.write(json.dumps(feedback_data) + "\n")
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
