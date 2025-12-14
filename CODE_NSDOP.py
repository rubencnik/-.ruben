import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os
import random
from PIL import Image
import time

# ========== SiameseNetwork (ТОЧНО ПО ВАШЕЙ СТРУКТУРЕ) ==========
class SiameseNetwork(tf.keras.Model):
    def __init__(self, input_shape=(160, 160, 3)):
        super(SiameseNetwork, self).__init__()
        
        # Построить embedding сеть
        self.backbone = self._build_backbone(input_shape)
        self.l2_normalize = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
     
    def _build_backbone(self, input_shape):
        """Построить backbone для извлечения embeddings"""
        model = tf.keras.Sequential([
            # Conv2D слои для извлечения признаков
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                         input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            
            # GlobalAveragePooling
            layers.GlobalAveragePooling2D(),
            
            # Dense слой для embedding (128-dimensional)
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)  # 128-мерное embedding
        ])
        return model
     
    def call(self, x1, x2, x3=None):
        """Forward pass"""
        # Вычислить embeddings
        e1 = self.backbone(x1)
        e2 = self.backbone(x2)
        
        # L2 нормализация
        e1 = self.l2_normalize(e1)
        e2 = self.l2_normalize(e2)
         
        if x3 is not None:
            e3 = self.backbone(x3)
            e3 = self.l2_normalize(e3)
            return e1, e2, e3
         
        return e1, e2

# ========== SiameseNetworkTrainer (ТОЧНО ПО ВАШЕЙ СТРУКТУРЕ) ==========
class SiameseNetworkTrainer:
    def __init__(self, siamese_net, learning_rate=0.0001):
        self.model = siamese_net
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     
    def contrastive_loss(self, e1, e2, y, margin=1.0):
        """Contrastive loss
        y=1 для identical images (хотим близость)
        y=0 для different images (хотим удаленность)
        Loss = y * d^2 + (1-y) * max(margin - d, 0)^2
        """
        # Евклидово расстояние
        distance = tf.sqrt(tf.reduce_sum(tf.square(e1 - e2), axis=1) + 1e-8)
        
        # Contrastive loss
        loss = y * tf.square(distance) + \
               (1 - y) * tf.square(tf.maximum(margin - distance, 0))
        
        return tf.reduce_mean(loss)
     
    def triplet_loss(self, e_anchor, e_positive, e_negative, margin=1.0):
        """Triplet loss
        Triplet loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
        d = Euclidean distance
        """
        # Вычисление расстояний
        pos_distance = tf.sqrt(tf.reduce_sum(tf.square(e_anchor - e_positive), axis=1) + 1e-8)
        neg_distance = tf.sqrt(tf.reduce_sum(tf.square(e_anchor - e_negative), axis=1) + 1e-8)
        
        # Triplet loss
        loss = tf.maximum(pos_distance - neg_distance + margin, 0.0)
        
        return tf.reduce_mean(loss)
     
    def train_step(self, x_anchor, x_positive, x_negative):
        """Один шаг обучения с triplet loss"""
        with tf.GradientTape() as tape:
            e_anchor, e_pos, e_neg = self.model(x_anchor, x_positive, x_negative)
            loss = self.triplet_loss(e_anchor, e_pos, e_neg, margin=1.0)
         
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
     
    def verify_faces(self, face1, face2, threshold=0.6):
        """Верифицировать два лица
        1. Получить embeddings
        2. Вычислить distance
        3. Если distance < threshold: same person
        """
        # Преобразуем в batch размером 1 если нужно
        if len(face1.shape) == 3:
            face1 = tf.expand_dims(face1, 0)
        if len(face2.shape) == 3:
            face2 = tf.expand_dims(face2, 0)
        
        # Получить embeddings
        e1, e2 = self.model(face1, face2)
        
        # Вычислить евклидово расстояние
        distance = tf.sqrt(tf.reduce_sum(tf.square(e1 - e2), axis=1) + 1e-8)
        distance_value = distance.numpy()[0]
        
        # Принять решение
        is_same = distance_value < threshold
        similarity = 1.0 - (distance_value / (threshold * 2))
        similarity = max(0.0, min(1.0, similarity))
        
        return bool(is_same), similarity, distance_value
     
    def identify_face(self, test_face, known_faces_embeddings, names, threshold=0.6):
        """Идентифицировать человека из тестового лица
        1. Получить embedding тестового лица
        2. Вычислить distances до всех known embeddings
        3. Найти minimum distance
        4. Если distance < threshold: вернуть имя
        Иначе: "Unknown"
        """
        # Получить embedding тестового лица
        if len(test_face.shape) == 3:
            test_face = tf.expand_dims(test_face, 0)
        
        # Получаем embedding через модель
        e_test = self.model.backbone(test_face)
        e_test = self.model.l2_normalize(e_test)[0]
        
        if not known_faces_embeddings:
            return "Unknown", float('inf'), []
        
        # Вычислить distances до всех known embeddings
        distances = []
        for emb in known_faces_embeddings:
            distance = tf.sqrt(tf.reduce_sum(tf.square(e_test - emb)) + 1e-8)
            distances.append(distance.numpy())
        
        distances = np.array(distances)
        
        # Найти минимальное расстояние и соответствующий индекс
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Принять решение
        if min_distance < threshold:
            return names[min_idx], min_distance, distances
        else:
            return "Unknown", min_distance, distances

# ========== FastDataGenerator (Новый, оптимизированный) ==========
class FastDataGenerator:
    """Быстрый генератор данных с предзагрузкой в память"""
    
    def __init__(self, data_dir='./face_data', img_size=(160, 160)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.people = {}
        self.load_all_images()
    
    def load_all_images(self):
        """Загружает все изображения в память один раз"""
        if not os.path.exists(self.data_dir):
            print(f"Создание папки {self.data_dir}...")
            os.makedirs(self.data_dir, exist_ok=True)
            self.create_demo_data()
        
        print(f"Загрузка данных из {self.data_dir}...")
        
        for person_name in os.listdir(self.data_dir):
            person_path = os.path.join(self.data_dir, person_name)
            if os.path.isdir(person_path):
                images = []
                for img_name in os.listdir(person_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, img_name)
                        img = Image.open(img_path).resize(self.img_size)
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        images.append(img_array)
                
                if images:
                    self.people[person_name] = images
        
        print(f"Загружено {len(self.people)} людей, "
              f"{sum(len(imgs) for imgs in self.people.values())} изображений")
    
    def create_demo_data(self):
        """Создает демо данные если папка пустая"""
        persons = ['Alice', 'Bob', 'Charlie']
        
        for person in persons:
            person_dir = os.path.join(self.data_dir, person)
            os.makedirs(person_dir, exist_ok=True)
            
            for i in range(5):
                # Создаем разноцветные изображения для разных людей
                if person == 'Alice':
                    color = [0.8, 0.2, 0.2]  # Красный
                elif person == 'Bob':
                    color = [0.2, 0.8, 0.2]  # Зеленый
                else:
                    color = [0.2, 0.2, 0.8]  # Синий
                
                img = np.ones((*self.img_size, 3)) * color
                img += np.random.normal(0, 0.1, img.shape)  # Добавляем шум
                img = np.clip(img, 0, 1) * 255
                img = img.astype(np.uint8)
                
                img_path = os.path.join(person_dir, f'img_{i}.jpg')
                Image.fromarray(img).save(img_path)
        
        print(f"Создано демо данных: {len(persons)} людей по 5 изображений")
    
    @tf.function
    def augment_image(self, image):
        """Аугментация изображения"""
        # Случайное отражение
        image = tf.image.random_flip_left_right(image)
        
        # Случайная яркость и контраст
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        return tf.clip_by_value(image, 0.0, 1.0)
    
    def generate_triplet_batch(self, batch_size=16):
        """Генерирует batch для triplet loss (ОЧЕНЬ БЫСТРО)"""
        person_names = list(self.people.keys())
        
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            # Выбираем человека для anchor и positive
            person = random.choice(person_names)
            person_images = self.people[person]
            
            # Anchor и positive (разные изображения одного человека)
            if len(person_images) >= 2:
                anchor_img, positive_img = random.sample(person_images, 2)
            else:
                anchor_img = person_images[0]
                positive_img = person_images[0].copy()
            
            # Negative (изображение другого человека)
            other_persons = [p for p in person_names if p != person]
            if other_persons:
                negative_person = random.choice(other_persons)
                negative_img = random.choice(self.people[negative_person])
            else:
                negative_img = anchor_img.copy() * 0.5  # Fallback
            
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
        
        # Преобразуем в тензоры
        anchors = tf.stack(anchors)
        positives = tf.stack(positives)
        negatives = tf.stack(negatives)
        
        # Аугментация целого батча
        anchors = self.augment_image(anchors)
        positives = self.augment_image(positives)
        negatives = self.augment_image(negatives)
        
        return anchors, positives, negatives

# ========== FaceRecognitionSystem (Простая система) ==========
class FaceRecognitionSystem:
    """Простая система распознавания лиц"""
    
    def __init__(self):
        self.model = SiameseNetwork(input_shape=(160, 160, 3))
        self.trainer = SiameseNetworkTrainer(self.model)
        self.data_gen = FastDataGenerator('./face_data')
        
        self.known_embeddings = []
        self.known_names = []
    
    def train(self, epochs=10, batch_size=16):
        """Обучение модели"""
        print("Начало обучения...")
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 50  # Количество batch за эпоху
            
            for batch in range(num_batches):
                x_anchor, x_positive, x_negative = self.data_gen.generate_triplet_batch(batch_size)
                batch_loss = self.trainer.train_step(x_anchor, x_positive, x_negative)
                epoch_loss += batch_loss
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss.numpy())
            
            print(f"Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Визуализация потерь
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        
        print("Обучение завершено!")
    
    def register_person(self, image_path, name):
        """Регистрация нового лица"""
        # Загружаем и предобрабатываем изображение
        img = Image.open(image_path).resize((160, 160))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = tf.expand_dims(img_array, 0)
        
        # Получаем embedding
        embedding = self.model.backbone(img_tensor)
        embedding = self.model.l2_normalize(embedding)[0]
        
        # Сохраняем
        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        
        print(f"Лицо '{name}' зарегистрировано")
    
    def verify(self, image1_path, image2_path, threshold=0.6):
        """Верификация двух лиц"""
        # Загружаем изображения
        img1 = Image.open(image1_path).resize((160, 160))
        img1_array = np.array(img1, dtype=np.float32) / 255.0
        
        img2 = Image.open(image2_path).resize((160, 160))
        img2_array = np.array(img2, dtype=np.float32) / 255.0
        
        # Верификация
        is_same, similarity, distance = self.trainer.verify_faces(img1_array, img2_array, threshold)
        
        # Отображение
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img1)
        axes[0].set_title('Изображение 1')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title('Изображение 2')
        axes[1].axis('off')
        
        result_color = 'green' if is_same else 'red'
        result_text = f"Один человек: {'ДА' if is_same else 'НЕТ'}\n"
        result_text += f"Сходство: {similarity:.1%}\n"
        result_text += f"Расстояние: {distance:.4f}\n"
        result_text += f"Порог: {threshold}"
        
        axes[2].text(0.1, 0.5, result_text, fontsize=12, color=result_color,
                    verticalalignment='center')
        axes[2].set_title('Результат')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return is_same, similarity
    
    def identify(self, test_image_path, threshold=0.6):
        """Идентификация лица"""
        # Загружаем изображение
        test_img = Image.open(test_image_path).resize((160, 160))
        test_img_array = np.array(test_img, dtype=np.float32) / 255.0
        
        if not self.known_embeddings:
            print("Нет зарегистрированных лиц!")
            return "Unknown", float('inf')
        
        # Идентификация
        name, distance, all_distances = self.trainer.identify_face(
            test_img_array, self.known_embeddings, self.known_names, threshold
        )
        
        # Отображение
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(test_img)
        axes[0].set_title('Тестовое изображение')
        axes[0].axis('off')
        
        result_text = f"Результат: {name}\n"
        result_text += f"Расстояние: {distance:.4f}\n"
        result_text += f"Порог: {threshold}"
        
        if name != "Unknown":
            result_text += f"\nУверенность: {max(0, 1 - distance/threshold):.1%}"
        
        axes[1].text(0.1, 0.5, result_text, fontsize=12,
                    verticalalignment='center')
        axes[1].set_title('Идентификация')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return name, distance

# ========== ОСНОВНОЙ ЗАПУСК ==========
def main():
    """Основная функция запуска"""
    print("=" * 60)
    print("SIAMESE NETWORK ДЛЯ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 60)
    
    # Инициализация системы
    system = FaceRecognitionSystem()
    
    # Обучение модели
    print("\n1. ОБУЧЕНИЕ МОДЕЛИ")
    system.train(epochs=5, batch_size=8)  # Быстрое обучение для демо
    
    # Регистрация известных лиц
    print("\n2. РЕГИСТРАЦИЯ ЛИЦ")
    
    # Регистрируем всех людей из данных
    for person_name, images in system.data_gen.people.items():
        # Берем первое изображение каждого человека
        person_dir = os.path.join('./face_data', person_name)
        img_files = [f for f in os.listdir(person_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            img_path = os.path.join(person_dir, img_files[0])
            system.register_person(img_path, person_name)
    
    # Тестирование верификации
    print("\n3. ТЕСТ ВЕРИФИКАЦИИ")
    
    # Получаем пути к изображениям для теста
    test_images = []
    for person_name in system.data_gen.people.keys():
        person_dir = os.path.join('./face_data', person_name)
        img_files = [f for f in os.listdir(person_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            img_path = os.path.join(person_dir, img_files[0])
            test_images.append((img_path, person_name))
    
    if len(test_images) >= 2:
        # Тест 1: Один и тот же человек
        img1_path, person1 = test_images[0]
        # Находим второе фото того же человека
        img2_path = None
        for img_path, person_name in test_images[1:]:
            if person_name == person1:
                img2_path = img_path
                break
        
        if img2_path:
            print(f"\nТест 1: Два фото одного человека ({person1})")
            is_same, similarity = system.verify(img1_path, img2_path, threshold=0.6)
            print(f"Результат: {'Правильно' if is_same else 'Ошибка'}, "
                  f"Сходство: {similarity:.1%}")
        
        # Тест 2: Разные люди
        if len(test_images) >= 3:
            img1_path, person1 = test_images[0]
            img2_path, person2 = test_images[1]
            if person1 != person2:
                print(f"\nТест 2: Два фото разных людей ({person1} и {person2})")
                is_same, similarity = system.verify(img1_path, img2_path, threshold=0.6)
                print(f"Результат: {'Ошибка' if is_same else 'Правильно'}, "
                      f"Сходство: {similarity:.1%}")
    
    # Тестирование идентификации
    print("\n4. ТЕСТ ИДЕНТИФИКАЦИИ")
    
    if test_images:
        test_img_path, expected_name = test_images[0]
        print(f"\nИдентификация: {os.path.basename(test_img_path)}")
        print(f"Ожидается: {expected_name}")
        
        name, distance = system.identify(test_img_path, threshold=0.6)
        print(f"Результат: {name}, Расстояние: {distance:.4f}")
        print(f"Правильно: {'Да' if name == expected_name else 'Нет'}")
    
    print("\n" + "=" * 60)
    print("СИСТЕМА ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
    print("=" * 60)
    
    return system

# Запуск системы
if __name__ == "__main__":
    try:
        system = main()
        
        print("\n" + "=" * 60)
        print("КОМАНДЫ ДЛЯ ИСПОЛЬЗОВАНИЯ:")
        print("=" * 60)
        print("1. Верификация двух лиц:")
        print("   system.verify('путь/к/изображению1.jpg', 'путь/к/изображению2.jpg')")
        print("\n2. Идентификация лица:")
        print("   system.identify('путь/к/тестовому_изображению.jpg')")
        print("\n3. Регистрация нового лица:")
        print("   system.register_person('путь/к/изображению.jpg', 'Имя')")
        print("\n4. Продолжить обучение:")
        print("   system.train(epochs=10, batch_size=16)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем.")
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Подсказка: Убедитесь, что папка './face_data' существует и содержит данные.")