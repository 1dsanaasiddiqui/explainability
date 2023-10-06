import numpy as np
import onnx
import onnxruntime
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

# Choose an image from the dataset (adjust the index as needed)
mnist_image_index = 0
image_to_explain = X_train[mnist_image_index]

# Display the original image
original_image_path = 'original_mnist_image.png'
Image.fromarray(image_to_explain).save(original_image_path)


# Flatten the image
flattened_image = image_to_explain.flatten()

# Load your ONNX model (adjust the path accordingly)
onnx_model_path = 'path_to_save_model.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)
onnx_model = onnx.load(onnx_model_path)

# LIME for MNIST
def generate_perturbations(num_perturbations=1000, img_size=(28, 28)):
    perturbations = np.random.normal(loc=0, scale=0.1, size=(num_perturbations, *img_size))
    return perturbations


def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def predict_model(model, instances):
    input_details = model.get_inputs()[0]

    # Print shapes for debugging
    print("Expected Input Shape:", input_details.shape)
    print("Input Instances Shape:", instances.shape)

    # Ensure the input shape matches the model's expected shape
    expected_shape = tuple(input_details.shape[1:])
    if instances.shape[1:] != expected_shape:
        raise ValueError(f"Invalid input shape. Expected {expected_shape}, got {instances.shape[1:]}.")

    return model.run(None, {input_details.name: instances.astype(np.float32)})[0]


def select_top_k(instances, predictions, k=5):
    #print(predictions)
    top_k_indices = np.argsort(predictions)[:k]
    return instances[top_k_indices]

def measure_fidelity(true_predictions, surrogate_predictions):
    return mean_squared_error(true_predictions, surrogate_predictions)

def train_interpretation_model(top_k_instances, true_labels):
    # Assuming top_k_instances is a NumPy array of perturbed instances
    flattened_instances = flatten_images(top_k_instances)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(flattened_instances, true_labels, test_size=0.2, random_state=42)

    # Create your interpretable model (e.g., logistic regression)
    interpretation_model = LogisticRegression()

    # Fit the interpretable model
    interpretation_model.fit(X_train, y_train)

    # Evaluate the model on the test set (optional)
    y_pred = interpretation_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Interpretation Model Accuracy: {accuracy}')

    return interpretation_model


def LIME(image, onnx_model, onnx_session, num_perturbations=1000, k=5):
    # Step 1: Generate Perturbed Instances
    perturbations = generate_perturbations(num_perturbations)

    # Step 2: Calculate Predictions for Perturbed Instances
    # Reshape image to match perturbations
    image_reshaped = image.reshape(1, 28, 28)
    perturbed_instances = image_reshaped + perturbations
    print(perturbed_instances)
    preds = []
    for img in perturbed_instances:
        perturbed_instances_flattened = flatten_images(img).reshape(1, -1)
        predictions = predict_model(onnx_session, perturbed_instances_flattened)
        preds.append(predictions)

    # Step 3: Select Top k Perturbed Instances
    top_k_instances = select_top_k(perturbed_instances, preds, k)
    #print(predictions)
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
    plt.close()  

# Apply LIME
interpretable_model = LIME(flattened_image, onnx_model, onnx_session)

# coefficients = interpretable_model.coef_.flatten()
# plt.bar(range(len(coefficients)), coefficients)
# plt.xlabel('Feature (Pixel)')
# plt.ylabel('Coefficient Value')
# plt.title('Interpretable Model Coefficients')

# # Save the interpretable model coefficients plot
# coefficients_plot_path = 'interpretable_model_coefficients.png'
# plt.savefig(coefficients_plot_path)
# plt.close()
print("Here")
