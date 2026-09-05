"""
Diagnostic: prints the actual architecture of brain_tumor_model.h5
so we know its real expected input size instead of guessing.
Run this once, read the output, then we fix classifier_agent.py to match.
"""
from tensorflow.keras.models import load_model

model = load_model("model/mobilenetv2_paper_exact.keras", compile=False)
print("\n=== MODEL INPUT SHAPE ===")
print(model.input_shape)
print("\n=== FULL MODEL SUMMARY ===")
model.summary()