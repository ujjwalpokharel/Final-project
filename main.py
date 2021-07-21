import cv2 as cv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
train_data=ImageDataGenerator(rescale=1./255)
train_generator=train_data.flow_from_directory(
    'computervision/data/train/',
    target_size=(300,300),
    batch_size=20,
    class_mode='binary'
)
validation_data=ImageDataGenerator(rescale=1./255)
validation_generator=validation_data.flow_from_directory(
    'computervision/data/validation/',
    target_size=(300,300),
    batch_size=14,
    class_mode='binary')
model.fit_generator(train_generator,
                    steps_per_epoch=10,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_steps=10,
                     verbose=2
                     )
cap=cv.VideoCapture(0)
while(True):
   value,frame=cap.read()
   cv.imshow('window',frame)
   if cv.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv.destroyAllWindows()
