import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


data_dir = 'C:\\dev\\workspaceMateus\\python_IA\\image_classifier\\teste4\\archive\\flowers'

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']




def open_image():
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)

            cv2.imshow('image asdasd', image)
            break

        break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_images_categories_memory():

    data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)

            try:
                # set image on cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.array(image, dtype=np.float32)

                # set image and label on data global
                data.append([image, label])

            except Exception as e:
                pass

    print(f"number of pictures: {len(data)}")

    # Pickle in Python is primarily used in serializing and deserializing a Python object structure.
    pik = open('data.pickle', 'wb')
    pickle.dump(data, pik)
    pik.close()


#load_images_categories_memory()


def load_data():

    # se data.pickle nao existe, executar o metodo antes: load_images_categories_memory

    pick = open('data.pickle', 'rb')
    data = pickle.load(pick)
    pick.close()

    #embaralhe
    np.random.shuffle(data)

    feature = []
    labels = []

    for img, label in data:
        feature.append(img)
        labels.append(label)

    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)

    feature = feature/255.0

    return [feature, labels]





