import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load your saved identity
reference_embedding = np.load("mrd_embedding.npy")

# Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Webcam
cap = cv2.VideoCapture(0)

THRESHOLD = 0.45

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding

        similarity = np.dot(reference_embedding, emb) / (
            np.linalg.norm(reference_embedding)
            * np.linalg.norm(emb)
        )

        x1, y1, x2, y2 = map(int, face.bbox)

        if similarity > THRESHOLD:
            label = f"MRD {similarity:.2f}"
            granted = True
        else:
            label = f"NOT MRD {similarity:.2f}"
            granted = False

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if granted else (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if granted else (0, 0, 255),
            2,
        )

        if granted:
            cv2.putText(
                frame,
                "ACCESS GRANTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
        else:
            cv2.putText(
                frame,
                "ACCESS DENIED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

    cv2.imshow("MRD Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()