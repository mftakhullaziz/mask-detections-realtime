# First step import library
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Define contruct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="path to input dataset")
ap.add_argument("-p","--plot", type=str, default="plot_train.png", help="path to output accuracy/loss")
ap.add_argument("-m", "--model", type=str,default="mask_detection_mobilenetV2.model", help="path to output face mask detector model")
args = vars(ap.parse_args())

# Inisialisasi Learning rate, epochs, dan Batch Size
Learning_rate = 1e-4
Epochs = 25
Bacth_size = 32

print("[INFO] Load data images .........")
Image_Paths = list(paths.list_images(args["dataset"]))
Data = []
Labels = []
for imagePath in Image_Paths :
    label = imagePath.split(os.path.sep)[-2]
    images = load_img(imagePath, target_size=(224,224))
    images = img_to_array(images)
    images = preprocess_input(images)

    Data.append(images)
    Labels.append(label)

Data = np.array(Data, dtype="float32")
Labels = np.array(Labels)
print("[INFO] Labels : ", label)

# One Hot Encoding Labels
Lb = LabelBinarizer()
Labels = Lb.fit_transform(Labels)
Labels = to_categorical(Labels)

# Pembagian Data
(trainX, testX, trainY, testY) = train_test_split(Data, Labels,
    test_size=0.10, stratify=Labels, random_state=42)

Data_Augmentation = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Train dengan MobileNet-V2
Base_model=MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224,224,3)))

models=Base_model.output
models=AveragePooling2D(pool_size=(3,3))(models)
models=Flatten(name="flatten")(models)
models=Dense(128, activation="relu")(models)
models=Dropout(0.5)(models)
models=Dense(2, activation="softmax")(models)


model = Model(inputs=Base_model.input, outputs=models)
for layer in Base_model.layers:
	layer.trainable = False

print("[INFO] compile models")
opt = Adam(lr=Learning_rate, decay=Learning_rate / Epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.summary()
print("[INFO] training models working")
H = model.fit(
	Data_Augmentation.flow(trainX, trainY, batch_size=Bacth_size),
	steps_per_epoch=len(trainX) // Bacth_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // Bacth_size,
	epochs=Epochs)

# Validation
print("[INFO] evaluating")
predIdxs = model.predict(testX, batch_size=Bacth_size)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=Lb.classes_))

print("[INFO] Save models")
model.save(args["model"], save_format="h5")

N = Epochs
plt.style.use("seaborn-white")
plt.figure()
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("Accuracy.png")

plt.style.use("seaborn-white")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("Loss.png")

