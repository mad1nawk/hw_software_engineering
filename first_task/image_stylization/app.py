import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import cv2



def load_image(img_path, target_size=None):
    """Загружает изображение, нормализует его (0, 1) и добавляет батч-размерность."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if target_size:
        img = tf.image.resize(img, target_size)

    return img[tf.newaxis, :]  # Возвращает тензор формы [1, H, W, 3]


def tensor_to_image(tensor):
    """Преобразует тензор обратно в стандартное изображение."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


# 1. Укажите пути к вашим изображениям
content_path = "new_cat.jpg"  # Что стилизуем (например, фото кота или города)
style_path = "van_gog.jpg"  # В какой стиль красим (например, картина Ван Гога)

# 2. Загружаем изображения
# Рекомендуемый размер для картинки стиля в этой модели — 256x256
content_image = load_image(content_path, target_size=(384, 384))
style_image = load_image(style_path, target_size=(256, 256))

print("Загрузка модели из TensorFlow Hub...")
# 3. Загружаем официальную модель Magenta от Google
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# 4. Выполняем перенос стиля (происходит мгновенно)
stylized_image_tensor = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# 5. Сохраняем результат
output_image = tensor_to_image(stylized_image_tensor)
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('stylized_result.jpg', output_image_bgr)

print("Готово! Результат сохранен в 'stylized_result.jpg'")

# 6. (Опционально) Отображаем результат, если запускаете в Jupyter/Colab
# plt.figure(figsize=(10, 10))
# plt.imshow(output_image)
# plt.axis('off')
# plt.show()
