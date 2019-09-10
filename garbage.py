import numpy as np
import pandas as pd
import keras
from keras.applications import ResNet50,VGG16
from keras.layers import Input, Dropout, Flatten, Dense, Activation
from keras import Model as keras_model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator

train_path="/kaggle/input/split-garbage-dataset/split-garbage-dataset/train"
valid_path="/kaggle/input/split-garbage-dataset/split-garbage-dataset/valid"
test_path="/kaggle/input/split-garbage-dataset/split-garbage-dataset/test"
'''
for dirname, _, filenames in os.walk(valid_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
epochs = 30 
batch_size = 64
print("hello")

# set model
inputs = Input(shape=[224, 224, 3])
# keras.backend.set_learning_phase(0)
# base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
# keras.backend.set_learning_phase(1)

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
for layer in base_model.layers[:-3]:
    layer.trainable = False 
X = base_model.output
fla = Flatten(name='flatten')(X)
fc1 = Dense(512, activation='relu')(fla)
drop1 = Dropout(0.2)(fc1)
#fc2 = Dense(1024, activation='relu')(fc1)
#drop2 = Dropout(0.5)(fc2)
fc3 = Dense(6, activation='softmax')(drop1)

model = keras_model(inputs=inputs, outputs=fc3)
model.summary()
# sgd=SGD(lr=0.1, decay=0.001, momentum=0.9, nesterov=True)
# nadam=Nadam(lr=0.0001)
adam=Adam(lr=0.00005)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# set callback
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10 ,verbose=1)

# data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)

# train model
model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    validation_data=valid_generator,
    validation_steps=10,
    epochs=epochs,
    #callbacks = [early_stopping],
    verbose=1
)

# model save
model.save_weights('./my_model_weights.h5') 

# model evaluate
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
scores = model.evaluate_generator(test_generator,STEP_SIZE_TEST)
print("test acc:",scores[1])
