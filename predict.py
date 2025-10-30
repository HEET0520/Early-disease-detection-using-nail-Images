import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2
import os
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO

# Import custom modules
from models import WeightedAvgFeatureExtractor
from data_utils import detect_and_crop, preprocess_image # Using the CV preprocessing

# --- Configuration ---
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")
BINARY_SVM_PATH = os.path.join(MODEL_DIR, "binary_svm_model.joblib")
MULTI_SVM_PATH = os.path.join(MODEL_DIR, "multiclass_svm_model.joblib")

# --- Disease Mapping (from cell 24) ---
disease_mapping = {
    0: "Acral Lentiginous Melanoma",
    1: "Bluish Nail",
    2: "Clubbing",
    3: "Healthy Nail", # Added for completeness
    4: "Koilonychia",
    5: "Median nail",
    6: "Nail Pitting",
    7: "Onychogryphosis",
    8: "Onychomycosis",
    9: "Yellow Nails"
}
HEALTHY_LABEL_INT = 3

# --- Load All Models Globally ---
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Feature Extractor
    feature_model = WeightedAvgFeatureExtractor().to(device)
    feature_model.eval() # Set to evaluation mode
    
    # 2. Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # 3. PCA
    pca_model = joblib.load(PCA_PATH)
    
    # 4. Classifiers
    svm_binary = joblib.load(BINARY_SVM_PATH)
    svm_multiclass = joblib.load(MULTI_SVM_PATH)

    print("All models loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run 'python train.py' to train and save the models.")
    feature_model = None

# --- Preprocessing for a single image ---
def preprocess_for_prediction(image_np):
    """
    Applies the CV preprocessing and then the PyTorch transforms
    for a single NumPy image.
    """
    preprocessed_cv = preprocess_image(image_np)
    if preprocessed_cv is None:
        raise ValueError("CV preprocessing failed.")
        
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(preprocessed_cv).unsqueeze(0)
    return image_tensor, preprocessed_cv # Return CV image for overlay

# --- Grad-CAM Generation (Based on cell 26 logic) ---
def generate_grad_cam(image_tensor, original_cv_image):
    """
    Generates a combined Grad-CAM heatmap from the three base models.
    """
    if feature_model is None:
        return None

    heatmaps = []
    
    # Target layers from your notebook
    target_layers = [
        (feature_model.base_model_1.features, '18'),
        (feature_model.base_model_2.features, '8'),
        (feature_model.base_model_3.features, '12') # '12' is the last conv block in SqueezeNet
    ]
    
    for model_features, layer_name in target_layers:
        gradients = []
        activations = []

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Find the target layer
        target_layer_found = None
        for name, module in model_features.named_modules():
            if layer_name in name:
                target_layer_found = module
                break
        
        if target_layer_found is None:
            print(f"Warning: Layer {layer_name} not found. Skipping heatmap.")
            continue

        # Register hooks
        f_hook = target_layer_found.register_forward_hook(forward_hook)
        b_hook = target_layer_found.register_backward_hook(backward_hook)

        # Forward pass
        image_tensor = image_tensor.to(device)
        model_features(image_tensor) # This triggers the forward hook
        
        # Get activations and clear list
        acts = activations[0].cpu()
        activations.clear()

        # Backward pass
        acts.retain_grad()
        # We backprop on the sum of the activations
        score = torch.sum(acts) 
        score.backward()
        
        # Get gradients and clear list
        grads = gradients[0].cpu()
        gradients.clear()
        
        # Remove hooks
        f_hook.remove()
        b_hook.remove()

        # Compute heatmap
        pooled_gradients = torch.mean(grads, dim=[0, 2, 3])
        for i in range(acts.shape[1]):
            acts[:, i, :, :] *= pooled_gradients[i]
        
        heatmap_tensor = torch.mean(acts, dim=1).squeeze()
        heatmap_tensor = nn.functional.relu(heatmap_tensor)
        
        # Normalize
        if torch.max(heatmap_tensor) > 0:
            heatmap_tensor /= torch.max(heatmap_tensor)
        
        heatmaps.append(heatmap_tensor.numpy())

    if not heatmaps:
        return None

    # Combine heatmaps by averaging
    fused_heatmap = np.mean(heatmaps, axis=0)
    
    # Resize heatmap and create overlay
    heatmap_resized = cv2.resize(fused_heatmap, (original_cv_image.shape[1], original_cv_image.shape[0]))
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_jet = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # Ensure original image is also 8-bit
    if original_cv_image.max() <= 1.0:
        original_cv_image = (original_cv_image * 255).astype(np.uint8)
    
    # Overlay
    superimposed_img = cv2.addWeighted(original_cv_image, 0.6, heatmap_jet, 0.4, 0)
    
    # Convert to base64 string to send to web
    _, img_encoded = cv2.imencode('.png', superimposed_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


# --- Main Prediction Function ---
def get_prediction(image_path):
    """
    Runs the full pipeline on a single image path and returns the prediction.
    """
    if feature_model is None:
        return "Error: Models are not loaded. Run train.py.", None, None, None

    try:
        # 1. Detect and Crop
        cropped_image = detect_and_crop(image_path)
        if cropped_image is None:
            return "Error: Could not process image (detect/crop failed).", None, None, None

        # 2. Preprocess and Convert to Tensor
        image_tensor, preprocessed_cv = preprocess_for_prediction(cropped_image)
        image_tensor = image_tensor.to(device)

        # 3. Extract Features
        with torch.no_grad():
            features = feature_model(image_tensor).cpu().numpy()
            if features.ndim == 1:
                features = features.reshape(1, -1) # Ensure 2D array
        
        # 4. Scale and Apply PCA
        scaled_features = scaler.transform(features)
        pca_features = pca_model.transform(scaled_features)
        
        # 5. Generate Grad-CAM (now that we have all inputs)
        grad_cam_image = generate_grad_cam(image_tensor, preprocessed_cv)
        
        # 6. Binary Classification
        binary_pred = svm_binary.predict(pca_features)[0]
        binary_proba = svm_binary.predict_proba(pca_features)[0]
        
        if binary_pred == 0:
            # Healthy (label 0 in binary model)
            confidence = binary_proba[0] # Probability of being class 0 (Healthy)
            return "Healthy Nail", {"Healthy Nail": confidence}, None, grad_cam_image
        else:
            # Diseased (label 1 in binary model)
            # 7. Multiclass Classification
            multi_pred = svm_multiclass.predict(pca_features)[0]
            multi_proba = svm_multiclass.predict_proba(pca_features)
            
            disease_name = disease_mapping.get(multi_pred, "Unknown Disease")
            
            # Create a dictionary of probabilities
            classes = svm_multiclass.classes_
            probabilities = multi_proba[0]
            prob_dict = {disease_mapping.get(cls): prob for cls, prob in zip(classes, probabilities) if cls in disease_mapping}
            
            # We can also pass back the PCA features for SHAP
            return disease_name, prob_dict, pca_features, grad_cam_image

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {e}", None, None, None

if __name__ == "__main__":
    # --- Test the prediction function ---
    TEST_IMAGE = "data/Bluish Nail_img8.jpg" # Example
    
    if not os.path.exists(TEST_IMAGE):
        print(f"Test image not found at '{TEST_IMAGE}'.")
    else:
        prediction, probabilities, _, grad_cam_img = get_prediction(TEST_IMAGE)
        print(f"\n--- Test Result for {TEST_IMAGE} ---")
        print(f"Prediction: {prediction}")
        if probabilities:
            print("Probabilities:")
            sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            for disease, prob in sorted_probs:
                print(f"  {disease}: {prob*100:.2f}%")
        
        if grad_cam_img:
            print("\nGrad-CAM image generated (base64 string, length):", len(grad_cam_img))