import joblib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFIG
FACES_DIR = 'static/faces'
MODEL_PATH = 'static/face_recognition_model.pkl'

def generate_confusion_matrix():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train it first.")
        return
    knn = joblib.load(MODEL_PATH)

    y_true = []
    y_pred = []

    # 2. Iterate through all user folders
    if not os.path.exists(FACES_DIR):
        print("Faces directory not found.")
        return

    users = os.listdir(FACES_DIR)
    
    print("Testing Model on saved images...")
    
    for user in users:
        user_path = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_path): continue

        # We will use the saved images as our "Test Set"
        # Ideally, you should split data, but for this assignment, 
        # testing on training data (or a subset) is acceptable for demonstration.
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            
            # Preprocess exactly like the real app
            resized_face = cv2.resize(img, (50, 50))
            flattened = resized_face.ravel().reshape(1, -1)
            
            # Predict
            prediction = knn.predict(flattened)[0]
            
            y_true.append(user)      # The folder name is the TRUE label
            y_pred.append(prediction) # The model's guess

    # 3. Create Confusion Matrix
    if len(y_true) == 0:
        print("No images found to test.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    
    # 4. Plot and Save
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Success! 'confusion_matrix.png' saved.")
    plt.show()

if __name__ == "__main__":
    generate_confusion_matrix()