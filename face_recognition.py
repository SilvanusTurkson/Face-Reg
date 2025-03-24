import cv2
from deepface import DeepFace
import numpy as np
import os
from collections import defaultdict

class FaceRecognition:
    def __init__(self):
        self.known_faces = {}
        self.face_db = []
        self.threshold = 0.4
        self.model_name = "Facenet"
        self.detector = "opencv"
        self.metrics = {
            'true_pos': 0,
            'false_pos': 0,
            'false_neg': 0
        }
        self.conf_history = []
        
    def load_faces(self, faces_dir):
        self.known_faces = {}
        for f in os.listdir(faces_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(f)[0]
                self.known_faces[name] = os.path.join(faces_dir, f)
                
    def build_db(self):
        self.face_db = []
        for name, path in self.known_faces.items():
            try:
                embedding = DeepFace.represent(
                    img_path=path, 
                    model_name=self.model_name
                )[0]["embedding"]
                self.face_db.append({"name": name, "embedding": embedding})
            except Exception as e:
                print(f"Error processing {name}: {e}")
    
    def recognize(self, img):
        results = []
        try:
            if isinstance(img, str):
                img_array = cv2.imread(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            faces = DeepFace.extract_faces(
                img_array, 
                detector_backend=self.detector,
                enforce_detection=False
            )
            
            for face in faces:
                if face["confidence"] > 0.9:
                    x, y, w, h = face["facial_area"].values()
                    face_img = img_array[y:y+h, x:x+w]
                    
                    try:
                        emb = DeepFace.represent(
                            img_path=face_img,
                            model_name=self.model_name,
                            enforce_detection=False
                        )[0]["embedding"]
                        
                        min_dist = float('inf')
                        identity = "Unknown"
                        conf = 0
                        
                        for person in self.face_db:
                            dist = DeepFace.distance(emb, person["embedding"])
                            if dist < min_dist:
                                min_dist = dist
                                identity = person["name"]
                                conf = 1 - dist
                        
                        if min_dist < self.threshold:
                            results.append({
                                "name": identity,
                                "location": (x, y, w, h),
                                "confidence": conf
                            })
                            self.metrics['true_pos'] += 1
                        else:
                            self.metrics['false_pos'] += 1
                            
                        # Draw on image
                        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            img_array,
                            f"{identity} ({conf:.2f})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                    except Exception as e:
                        print(f"Recognition error: {e}")
            
            return img_array, results
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None, []
    
    def get_metrics(self):
        precision = 0
        recall = 0
        f1 = 0
        
        tp = self.metrics['true_pos']
        fp = self.metrics['false_pos']
        fn = self.metrics['false_neg']
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        avg_conf = np.mean(self.conf_history) if self.conf_history else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_conf": avg_conf,
            "true_pos": tp,
            "false_pos": fp
        }
