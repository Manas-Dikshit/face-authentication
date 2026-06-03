import cv2
import numpy as np
from insightface.app import FaceAnalysis

REFERENCE = "mrd_embedding.npy"
TEST_IMAGE = "test.jpg"   # Change this

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

reference_embedding = np.load(REFERENCE)

img = cv2.imread(TEST_IMAGE)

if img is None:
    raise Exception("Image not found")

faces = app.get(img)

if len(faces) == 0:
    raise Exception("No face detected")

embedding = faces[0].embedding

# Cosine similarity
similarity = np.dot(reference_embedding, embedding) / (
    np.linalg.norm(reference_embedding)
    * np.linalg.norm(embedding)
)

print(f"Similarity: {similarity:.4f}")

if similarity > 0.45:
    print("✅ ACCESS GRANTED (MRD)")
else:
    print("❌ ACCESS DENIED")