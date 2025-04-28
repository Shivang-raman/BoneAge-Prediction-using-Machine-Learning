# BoneAge-Prediction-using-Machine-Learning
Automating pediatric bone age assessment using deep learning for faster and more accurate clinical diagnosis

Welcome to the **Bone Age Prediction** project!  
This repository provides a machine learning-based solution to predict the bone age of pediatric patients from hand X-ray images. Accurate bone age estimation is crucial in diagnosing growth disorders and planning treatments.

## 📌 Project Overview

Bone age prediction plays a vital role in pediatric radiology by assessing the maturity of bones compared to chronological age. This project applies deep learning techniques on radiographic images to automate bone age assessment, aiming to:

- Improve prediction accuracy
- Reduce radiologist workload
- Assist in early diagnosis of growth abnormalities

---

## ⚙️ Project Details

- **Language:** Python 3.12
- **Platform:** Google Colab
- **Libraries:** TensorFlow, Keras, Scikit-learn, OpenCV, Matplotlib, NumPy, Pandas
- **Dataset:** Bone X-ray images with corresponding bone ages (from open sources like RSNA, or customized CSV-based datasets)
- **Storage:** Google Drive integration for dataset handling and model saving
- **Task Type:** Regression (continuous bone age prediction)

## 🛠️ Features

- 📈 Data preprocessing (image resizing, normalization)
- 🧠 Model development (Convolutional Neural Networks - CNNs)
- 🔥 Transfer Learning (using pretrained models for faster convergence)
- 🎯 Evaluation metrics: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- 🏆 Hyperparameter tuning and Cross-validation
- 💾 Model saving and loading via Google Drive
- 📊 Visualization of training progress and results

## 🚀 Installation and Setup

Follow these steps to set up and run the project:

1. **Clone the repository**  
   git clone https://github.com/your-username/BoneAge_Prediction.git
   cd BoneAge_Prediction

2. Install required libraries
   Make sure you have Python 3.12 installed.
   !pip install tensorflow keras scikit-learn opencv-python matplotlib numpy pandas

3. Set up Google Drive

   Mount your Google Drive in Google Colab for loading datasets and saving models.
     
4. Run the Notebook
   Open BoneAge_Prediction.ipynb in Google Colab and execute cells step-by-step.

🧪 Model Training Workflow
1. Load the dataset (X-ray images + CSV labels)

2. Preprocess data (resizing, scaling, augmentations if necessary)

3. Split data into Train/Validation/Test sets

4. Build a CNN or use a pretrained model (e.g., ResNet, EfficientNet)

5. Compile model with optimizer, loss function (Mean Squared Error), and metrics

6. Train model and monitor performance

7. Evaluate model on the test set

8. Save the model to Google Drive

📊 Results and Performance
*****The Results are Clearly mentioned in the notebook******

✨ Future Improvements
Implement ensemble models to boost performance

Apply advanced augmentation techniques (CutMix, MixUp)

Deploy the model via a web application (Flask/Django)

Incorporate explainability (Grad-CAM visualization)

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

**********Some Major Snapshots********************

![download (2)](https://github.com/user-attachments/assets/ee9deb41-7165-4d85-bf72-a9136a8c01b2)

![download (3)](https://github.com/user-attachments/assets/933ec95e-1c88-468f-9a98-f1452b8682b8)

![download (4)](https://github.com/user-attachments/assets/73aba3a7-c482-4356-846d-a591a9d5e9e9)

![download (5)](https://github.com/user-attachments/assets/9270f9e9-6eff-4054-b7f0-5582daff25fc)

![download (6)](https://github.com/user-attachments/assets/48752191-2a5b-4ee9-8acb-eee8fa8d170f)

![download (7)](https://github.com/user-attachments/assets/cc86888f-9126-449e-9b50-2ac6f79e51f1)

![download (8)](https://github.com/user-attachments/assets/31123b42-38cb-4b27-bc76-3849eb1ca737)

