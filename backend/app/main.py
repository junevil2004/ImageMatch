
import os
import json
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid

# --- Configuration ---
app = FastAPI()
STORAGE_PATH = "D:/_project/ImageMatch/storage"
IMAGE_STORAGE_PATH = os.path.join(STORAGE_PATH, "images")
VECTOR_STORAGE_PATH = os.path.join(STORAGE_PATH, "vectors.json")
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
async def search_similar_images(file: UploadFile = File(...), top_k: int = 5):
    try:
        # Save the query image temporarily
        query_image_path = os.path.join(IMAGE_STORAGE_PATH, f"query_{file.filename}")
        with open(query_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Generate the vector for the query image
        query_vector = generate_vector(query_image_path)
        os.remove(query_image_path) # Clean up the temporary query image

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

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/gallery/")
def get_gallery():
    try:
        vectors = load_vectors()
        return {"images": list(vectors.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
