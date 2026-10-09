import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2


def load_image(img_path, target_size=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if target_size:
        img = tf.image.resize(img, target_size)

    return img[tf.newaxis, :]


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

content_path = "new_cat.jpg"
style_path = "van_gog.jpg"

content_image = load_image(content_path, target_size=(384, 384))
style_image = load_image(style_path, target_size=(256, 256))

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

stylized_image_tensor = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

output_image = tensor_to_image(stylized_image_tensor)
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('stylized_result.jpg', output_image_bgr)
