import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import onnx
import onnxruntime
import matplotlib.pyplot as plt
import ast

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def reshape_images(flattened_images):
    return flattened_images.reshape(1000, 28, 28)

def generate_perturbations(x, num_perturbations=1000):
    perturbations = np.random.normal(loc=0, scale=0.1, size=(num_perturbations, 28, 28))
    return perturbations

def predict_model(model, instances):
    input_details = model.get_inputs()[0]
    instances = instances.reshape(-1, *input_details.shape[1:])
    return model.run(None, {input_details.name: instances.astype(np.float32)})[0]


def select_top_k(instances, predictions, k=5):
    top_k_indices = np.argsort(predictions)[:k]
    return instances[top_k_indices]

def measure_fidelity(true_predictions, surrogate_predictions):
    return mean_squared_error(true_predictions, surrogate_predictions)

def train_interpretation_model(instances, true_model):
    interpretation_model = LinearRegression()
    flattened_instances = flatten_images(instances)
    interpretation_model.fit(flattened_instances, true_model)
    return interpretation_model

def LIME(image, true_model, onnx_session, num_perturbations=1000, k=5):
    # Step 1: Generate Perturbed Instances
    perturbations = generate_perturbations(image, num_perturbations)

    # Step 2: Calculate Predictions for Perturbed Instances
    img = reshape_images(np.array(image))
    perturbed_instances = img + perturbations
    perturbed_instances_flattened = flatten_images(perturbed_instances)
    predictions = predict_model(onnx_session, perturbed_instances_flattened)

    # Step 3: Select Top k Perturbed Instances
    top_k_instances = select_top_k(perturbed_instances, predictions, k)

    # Step 4: Train Interpretation Model
    interpretation_model = train_interpretation_model(top_k_instances, onnx_model)

    return interpretation_model

def save_interpretable_model_plot(interpretable_model, feature_names, save_path):
    # Extract coefficients and feature names
    coefficients = interpretable_model.coef_
    
    # Plot the coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(coefficients)), coefficients)
    plt.xticks(range(len(coefficients)), feature_names, rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title('Interpretable Model Coefficients')
    
    # Save the plot as an image
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up resources

# Example usage
# Assume 'X' is your dataset
import onnx

# Load your ONNX model
onnx_model_path = 'path_to_save_model.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)
onnx_model = onnx.load(onnx_model_path)

# Create an ONNX Runtime Inference Session
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

f = open("../images/3img9483", "r")
image = ast.literal_eval(f.read())

# Choose an instance to explain
instance_to_explain = image

# Apply LIME
interpretable_model = LIME(image, onnx_model, onnx_session)
save_path = 'interpretable_model_plot.png'


save_interpretable_model_plot(interpretable_model, feature_names, save_path)

# Now you can use 'interpretable_model' for interpretation purposes.
