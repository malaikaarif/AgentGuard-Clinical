"""
Explainability agent — generates a Grad-CAM heatmap overlay for a
single image, using the exact same technique from the Colab notebook
that produced MedTrust-Audit's sample heatmaps.

This is the piece the reasoning agent's claims get checked against:
if the reasoning agent said "the sella turcica and suprasellar region
support this diagnosis," this heatmap tells us where the model
actually looked. Comparing those two is phase 3's most important
step, not yet done here — this module just produces the heatmap.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

IMG_SIZE = (224, 224)
MODEL_PATH = "model/mobilenetv2_paper_exact.keras"
OUTPUT_DIR = "explainability_output"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
    return _model


def _find_last_conv_layer_name(model) -> str:
    """Same logic as the Colab notebook: walk layers backward, return
    the first Conv2D (or name containing 'conv')."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def _make_gradcam_heatmap(img_array, model, last_conv_layer_name) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(img_path: str, heatmap: np.ndarray, alpha: float = 0.4):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)

    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    return array_to_img(superimposed)


def generate_explainability(image_path: str) -> dict:
    """
    Runs Grad-CAM on a single image and saves the overlay.

    Returns:
        {
            "heatmap_path": str,        # where the overlay PNG was saved
            "heatmap_array": np.ndarray # raw heatmap, in case later code
                                         # needs to compute region overlap
                                         # against the reasoning agent's claims
        }
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = _get_model()

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    last_conv_layer_name = _find_last_conv_layer_name(model)
    heatmap = _make_gradcam_heatmap(arr, model, last_conv_layer_name)
    overlay = _overlay_heatmap(image_path, heatmap)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam.png")
    overlay.save(save_path)

    return {
        "heatmap_path": save_path,
        "heatmap_array": heatmap,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python explainability_agent.py <path_to_image>")
        sys.exit(1)

    result = generate_explainability(sys.argv[1])
    print(f"Saved Grad-CAM overlay to: {result['heatmap_path']}")