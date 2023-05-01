import coremltools as ct

# Load the TensorFlow 2 SavedModel
saved_model_path = "saved_model"

# Convert the SavedModel to Core ML format
coreml_model = ct.convert(saved_model_path, source='tensorflow')

# Save the Core ML model
coreml_model.save("Inpainting.mlmodel")
