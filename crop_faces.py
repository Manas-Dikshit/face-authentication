import os
import cv2
from insightface.app import FaceAnalysis

INPUT_DIR = "my_photos"  # Change this to your folder with photos
OUTPUT_DIR = "cropped_faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)
    if img is None:
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"No face found: {file}")
        continue

    # Keep the largest detected face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    x1, y1, x2, y2 = map(int, face.bbox)

    pad = 30
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    crop = img[y1:y2, x1:x2]

    cv2.imwrite(os.path.join(OUTPUT_DIR, file), crop)

print("Done")