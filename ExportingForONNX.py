import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib


# Load your best model
best_model = joblib.load("models\knn.pkl")  # right now, the best is KNN (23rd of May 2025)

# Define input shape: (None, feature_dim), here assuming 34 features
initial_type = [('input', FloatTensorType([None, 34]))]

# Convert to ONNX
onnx_model = convert_sklearn(best_model, initial_types=initial_type)

# Save to file
with open("best_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model exported to ONNX format as best_model.onnx")
