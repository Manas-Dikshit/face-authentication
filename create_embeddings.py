import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

FACE_DIR = "cropped_faces"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

embeddings = []

for file in os.listdir(FACE_DIR):
    path = os.path.join(FACE_DIR, file)

    img = cv2.imread(path)
    if img is None:
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"No embedding: {file}")
        continue

    embeddings.append(faces[0].embedding)

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

# Create your identity vector
reference_embedding = embeddings.mean(axis=0)

np.save("mrd_embedding.npy", reference_embedding)

print("Saved: mrd_embedding.npy")