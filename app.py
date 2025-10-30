from flask import Flask, request, render_template, send_from_directory
import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Import your prediction function and loaded models
from predict import get_prediction, svm_multiclass, pca_model, disease_mapping, feature_model
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a-very-secret-key-for-flask' # Change this

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- SHAP Explainer (Loaded once) ---
print("Creating SHAP explainer background...")
try:
    if feature_model is not None:
        dummy_features = np.zeros((10, 512))
        dummy_scaled = joblib.load(os.path.join('models', 'scaler.joblib')).transform(dummy_features)
        background_pca = joblib.load(os.path.join('models', 'pca_model.joblib')).transform(dummy_scaled)
        
        shap_explainer = shap.KernelExplainer(svm_multiclass.predict_proba, background_pca)
        print("SHAP explainer created successfully.")
    else:
        shap_explainer = None
        print("Models not loaded, SHAP explainer skipped.")
except Exception as e:
    shap_explainer = None
    print(f"Could not create SHAP explainer: {e}")


def get_shap_plot(pca_features, probabilities):
    if shap_explainer is None or pca_features is None:
        return None
    try:
        print("Generating SHAP plot...")
        shap_values = shap_explainer.shap_values(pca_features)
        
        top_pred_index = np.argmax(list(probabilities.values()))
        top_pred_class = list(probabilities.keys())[top_pred_index]
        top_pred_class_idx = svm_multiclass.classes_[top_pred_index]
        
        print(f"Top predicted class for SHAP: {top_pred_class} (index {top_pred_class_idx})")

        fig = shap.force_plot(
            shap_explainer.expected_value[top_pred_index],
            shap_values[top_pred_index][0], 
            feature_names=[f"PCA_Feat_{i}" for i in range(pca_features.shape[1])],
            matplotlib=True,
            show=False
        )
        
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        print("SHAP plot generated.")
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None


# --- HTML Templates ---
template_index = """
<!doctype html>
<title>Nail Disease Classifier</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #212529; }
  h1 { color: #343a40; }
  form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
  input[type=file] { margin-right: 10px; }
  input[type=submit] { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; font-weight: bold; }
  input[type=submit]:hover { background: #0056b3; }
</style>
<h1>Upload a Nail Image for Classification</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/png, image/jpeg">
  <input type=submit value=Upload>
</form>
"""

template_result = """
<!doctype html>
<title>Prediction Result</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #212529; }
  h1, h2, h3 { color: #343a40; }
  h2.healthy { color: #28a745; }
  h2.diseased { color: #dc3545; }
  .container { display: flex; flex-wrap: wrap; gap: 40px; }
  .result-text { max-width: 500px; }
  img.main-img { max-width: 400px; max-height: 400px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
  ul { list-style-type: none; padding-left: 0; }
  li { background: #fff; margin-bottom: 5px; padding: 8px 12px; border-radius: 4px; border: 1px solid #dee2e6; }
  li strong { color: #495057; }
  a { color: #007bff; text-decoration: none; font-weight: bold; }
  a:hover { text-decoration: underline; }
  .plots-container { margin-top: 30px; }
  .shap-plot img { max-width: 100%; height: auto; border: 1px solid #ccc; }
  .gradcam-plot img { max-width: 400px; height: auto; border: 1px solid #ccc; border-radius: 8px; }
</style>
<h1>Prediction Result</h1>
<div class="container">
  <div class="result-image">
    <h3>Uploaded Image:</h3>
    <img class="main-img" src="{{ url_for('uploaded_file', filename=filename) }}">
  </div>
  <div class="result-text">
    <h2 class="{{ 'healthy' if prediction == 'Healthy Nail' else 'diseased' }}">
      Prediction: {{ prediction }}
    </h2>
    {% if probabilities %}
      <h3>Confidence Scores:</h3>
      <ul>
      {% for disease, prob in probabilities.items()|sort(attribute='1', reverse=True) %}
        <li><strong>{{ disease }}:</strong> {{ "%.2f"|format(prob*100) }}%</li>
      {% endfor %}
      </ul>
    {% endif %}
    <br><br>
    <a href="/">Upload another image</a>
  </div>
</div>
<div class="plots-container">
  {% if gradcam_image %}
    <div class="gradcam-plot">
      <h3>Feature Extractor Heatmap (Grad-CAM)</h3>
      <p>This shows which parts of the image the deep learning model focused on.</p>
      <img src="{{ gradcam_image }}" alt="Grad-CAM Heatmap">
    </div>
  {% endif %}
  {% if shap_image %}
    <div class="shap-plot">
      <h3>Classifier Explanation (SHAP)</h3>
      <p>This plot shows which features contributed most to the prediction of <strong>{{ prediction }}</strong>.</p>
      <img src="{{ shap_image }}" alt="SHAP Plot">
    </div>
  {% endif %}
</div>
"""

# --- App Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction, probabilities, PCA features, and Grad-CAM
            prediction, probabilities, pca_features, grad_cam_image = get_prediction(filepath)
            
            # Generate SHAP plot only if it's a disease
            shap_image = None
            if prediction != "Healthy Nail" and probabilities is not None:
                shap_image = get_shap_plot(pca_features, probabilities)
            
            return render_template(
                'result.html', 
                prediction=prediction, 
                probabilities=probabilities,
                filename=filename,
                shap_image=shap_image,
                gradcam_image=grad_cam_image
            )
            
    # For a GET request, just show the upload form
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create the template files dynamically
    if not os.path.exists('templates'):
        os.makedirs('templates')
    with open('templates/index.html', 'w') as f:
        f.write(template_index)
    with open('templates/result.html', 'w') as f:
        f.write(template_result)
    
    # Run the app
    print("Flask app starting... Open http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True, host='127.0.0.1', port=5000)