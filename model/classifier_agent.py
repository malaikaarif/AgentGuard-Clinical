"""
Classifier agent — wraps the trained MobileNetV2 brain tumor model
(from Beyond Accuracy / MedTrust-Audit) as a single callable function
for use as the first agent in the AgentGuard-Clinical pipeline.
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Must match training exactly (see Colab training cell)
IMG_SIZE = (224, 224)

# Order matters — this must match train_generator.class_indices from training,
# which is alphabetical by folder name. Confirm this against your training
# notebook output ("Classes: [...]") before trusting it blindly.
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Path to the trained model file. Point this at wherever you place the
# .h5 or .keras file inside model/ — do NOT commit this file to git.
MODEL_PATH = "model/mobilenetv2_paper_exact.keras"

_model = None  # lazy-loaded singleton so the model loads once, not per-call


def _get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
    return _model


def classify_image(image_path: str) -> dict:
    """
    Takes a path to a single image, runs it through the trained
    MobileNetV2 classifier, and returns diagnosis + confidence + raw logits.

    Returns:
        {
            "diagnosis": str,        # predicted class name
            "confidence": float,     # softmax probability of predicted class
            "logits": list[float],   # full softmax output vector, all 4 classes
            "class_names": list[str] # for reference, order matches logits
        }
    """
    model = _get_model()

    # Preprocessing must exactly match training: resize to 224x224,
    # then MobileNetV2's ImageNet-style preprocess_input (NOT plain /255).
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)          # add batch dimension
    arr = preprocess_input(arr)

    probs = model.predict(arr, verbose=0)[0]   # shape: (4,)
    pred_idx = int(np.argmax(probs))

    return {
        "diagnosis": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "logits": probs.tolist(),
        "class_names": CLASS_NAMES,
    }


if __name__ == "__main__":
    # Quick standalone test — replace with a real path to one of your test images
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classifier_agent.py <path_to_image>")
        sys.exit(1)

    result = classify_image(sys.argv[1])
    print(result)