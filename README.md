# 🛸 Aerial Surveillance: Drone vs 🦅 Bird Object Classification & Detection

## 📌 Project Overview
This project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone and optionally perform object detection to locate and label these objects in real-world scenes.
The solution will help in security surveillance, wildlife protection, and airspace safety where accurate identification between drones and birds is critical. The project involves building a Custom CNN classification model, leveraging transfer learning, and optionally implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.

📌 Real-Time Business Use Cases

Wildlife Protection
Detect birds near wind farms or airports to prevent accidents.
Security & Defense Surveillance
Identify drones in restricted airspace for timely alerts.
Airport Bird-Strike Prevention
Monitor runway zones for bird activity.
Environmental Research
Track bird populations using aerial footage without misclassification.

## Model Building (Classification)
Custom CNN: Conv layers, pooling, dropout, batch normalization, dense output layer
Transfer Learning: Load models like ResNet50, MobileNet, EfficientNetB0 and fine-tune

## Model Training
Train both models
Use EarlyStopping & ModelCheckpoint
Track metrics: Accuracy, Precision, Recall, F1-score

## Model Evaluation
Evaluate test results with confusion matrix & classification report
Plot accuracy/loss graphs

🛑 Early Stopping triggered at epoch 10!

📊 Custom_CNN Final Classification Report:
              precision    recall  f1-score   support

        bird       0.95      0.56      0.70        36
       drone       0.53      0.95      0.68        19

    accuracy                           0.69        55
   macro avg       0.74      0.75      0.69        55
weighted avg       0.81      0.69      0.69        55

Confusion Matrix:
[[20 16]
 [ 1 18]]
<img width="611" height="390" alt="download" src="https://github.com/user-attachments/assets/ad022c6c-01f9-414c-8b16-142cd50ae6b0" />

🛑 Early Stopping triggered at epoch 6!

📊 ResNet50_Transfer Final Classification Report:
              precision    recall  f1-score   support

        bird       0.96      0.72      0.83        36
       drone       0.64      0.95      0.77        19

    accuracy                           0.80        55
   macro avg       0.80      0.83      0.80        55
weighted avg       0.85      0.80      0.80        55

Confusion Matrix:
[[26 10]
 [ 1 18]]
 <img width="617" height="390" alt="download" src="https://github.com/user-attachments/assets/45593cca-bc8e-4185-a3da-2d996338d6c4" />

## 📂 Repository Structure
```text
Project_5/
├── classification_Dataset/
│   ├── test/                 
│   ├── train/                 
│   └── val/                  
├── train/
│   └── bird and drone

├── app.py                     # Streamlit application UI script

```
Project_5/
├── object_detection_Dataset/
│   ├── train/                 # Balanced training image tensors & .txt vectors
│   ├── valid/                 # Control split for calculating mAP
│   └── test/                  # Pure unseen data for testing inference
├── runs/
│   └── detect/
│       └── train/weights/     # Houses your final trained 'best.pt' weights
├── data.yaml                  # Class mappings (0: bird, 1: drone) & root anchors
├── app_yolo.py                     # Streamlit application UI script
└── yolo_pipeline.py                # Training execution code block
```
## 📊 Dataset Configuration (`data.yaml`)
The project utilizes targeted bounding box parameters to enforce class mapping:
```yaml
path: C:/Users/ThinkPad/Project_5/object_detection_Dataset
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: bird
  1: drone
```
## 🛠️ Performance & Training Optimization
To deploy this architecture locally on standard **CPU hardware**, the model utilizes specific mathematical runtime optimizations:
* **Feature Resolution Drop (`imgsz=416`)**: Balanced resolution matrix to retain fine geometric lines of propellers without bottle-necking CPU processing loops.
* **Aggressive Batch Optimization**: Lower batch sizes to maintain a steady memory footprint and eliminate background runtime overhead.
* **Early Convergence Checks**: Employs structural patience parameters to cease epochs once validation loss flattens.

---

## 🚀 Installation & Local Deployment

### 1. Clone the Project
```bash
git clone https://github.com
cd Project_5
```

### 2. Set Up Environment & Dependencies
Ensure you have Python 3.8+ installed, then download the core processing frameworks:
```bash
pip install ultralytics streamlit opencv-python-headless
```

### 3. Run the Custom GUI Terminal
Launch your web-based machine learning workspace locally:
```bash
streamlit run app.py
<img width="720" height="357" alt="Capture1" src="https://github.com/user-attachments/assets/f6cfb144-36e3-4423-9fc9-3f742fcd08be" />
<img width="646" height="411" alt="Capture2" src="https://github.com/user-attachments/assets/04e91687-f5e4-4c0b-8885-8d54a083c615" />
<img width="690" height="437" alt="Capture3" src="https://github.com/user-attachments/assets/edfb1088-2ba6-4521-b944-6bfacbe44161" />

streamlit run app_yolo.py

<img width="892" height="358" alt="yolo1" src="https://github.com/user-attachments/assets/cd1ed88d-ae9b-438d-a51c-8c56d7f4d90d" />

<img width="910" height="390" alt="yolo2" src="https://github.com/user-attachments/assets/b827ae43-2870-400b-9ae2-d91518d85628" />

<img width="921" height="426" alt="yolo3" src="https://github.com/user-attachments/assets/2e4d6e44-0134-42e4-8314-ad5b1dfa1e42" />



```
Open `http://localhost:8501` in your browser, adjust your **Confidence Slider** parameters down to isolate micro-targets, and verify performance on unseen `test/images` directory frames.

## 📝 Observations & Key Insights

* **Dataset Balance**: To ensure optimal model fairness, data must never be skewed. The training pipeline utilizes a perfectly equal distribution of bird and drone images to eliminate classification bias.
* **Epoch Optimization**: Training is set to **30 epochs or higher**. This gives the network sufficient training cycles to clearly distinguish and identify the fine structural features of both classes.
* **Resolution Control**: Increasing image resolution (`imgsz=416`) is critical. Higher pixel density preserves sharp geometric lines (like drone propellers), preventing the model from confusing distant objects.
* **Data Scale**: Deep learning models thrive on scale. To improve real-world reliability and deployment accuracy, the training corpus should continually be expanded with a larger, more diverse image dataset.

## 📈 Future Roadmaps
- Integrate **DeepSORT object tracking algorithms** to track directional vector pathways of drones in high-speed sequences.
- Expand edge-computing compatibility for live deployments directly inside standalone embedded drone camera arrays.
