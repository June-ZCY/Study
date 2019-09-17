
import cv2 as cv
from cv2 import imread
import random
import re
from tqdm import tqdm

image_path="/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png/"
mask_path="/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/masks/"
images=os.listdir(image_path)
mask=os.listdir(mask_path)

# Y_train_file=random.sample(mask ,550)
# Y_test_file=list(set(mask)-set(Y_train_file))

# X_train_file=[]
# X_test_file=[]

# for i in Y_train_file:
#     if "_mask" in i:
#         i=re.sub("_mask", "", i)
#     X_train_file.append(i)

    
# for i in Y_test_file:
#     if "_mask" in i:
#         i=re.sub("_mask", "", i)
#     X_test_file.append(i)

Y_file=mask
X_file=[]
for i in Y_file:
    if "_mask" in i:
        i=re.sub("_mask", "", i)
    X_file.append(i)


# def setdata(flag):
#     X=[]
#     Y=[]
#     if flag=="train":
#         for i in tqdm(X_train_file): 
#             x = cv.resize(cv.imread(os.path.join(image_path,i)),(512,512))[:,:,0]
#             X.append(x)
#         for j in tqdm(Y_train_file): 
#             y = cv.resize(cv.imread(os.path.join(mask_path,j)),(512,512))[:,:,0]
#             Y.append(y)

#     if flag=="test":
#         for i in tqdm(X_test_file): 
#             x = cv.resize(cv.imread(os.path.join(image_path,i)),(512,512))[:,:,0]
#             X.append(x)
#         for j in tqdm(Y_test_file): 
#             y = cv.resize(cv.imread(os.path.join(mask_path,j)),(512,512))[:,:,0]
#             Y.append(y)

#     return X,Y
def setdata():
    X=[]
    Y=[]
    for i in tqdm(X_file): 
        x = cv.resize(cv.imread(os.path.join(image_path,i)),(512,512))[:,:,0]
        X.append(x)
    for j in tqdm(Y_file): 
        y = cv.resize(cv.imread(os.path.join(mask_path,j)),(512,512))[:,:,0]
        Y.append(y)

    return X,Y
X_data, Y_data = setdata()

# X_train, Y_train = setdata("train")
# X_test, Y_test = setdata("test")



import numpy as np
# print(len(X_train))
# print(X_train[0].shape)

# print(type(X_train[0][0][0]))
print(len(X_data))
print(X_data[0].shape)



import matplotlib.pyplot as plt
plt.subplot(4,6,1)
j=0
for i in range(6):
    j+=1
    plt.subplot(3,4,j)
    plt.imshow(X_data[i])
    j+=1
    plt.subplot(3,4,j)
    plt.imshow(Y_data[i])



X_data = np.array(X_data).reshape(len(X_data),512,512,1)
Y_data = np.array(Y_data).reshape(len(Y_data),512,512,1)

print(X_data.shape)
print(len(X_data))
print(X_data[0].shape)

X_data = np.concatenate(X_data,axis=0)    
Y_data = np.concatenate(Y_data,axis=0)   
print(X_data.shape)

from keras.models import Model
from keras.layers import *
from keras.optimizers import SGD, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def Unet(input_size = (512,512,1)):
    inputs = Input(input_size)
    #下采样
    down1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(inputs)
    down1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(down1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)
    
    down2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(pool1)
    down2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(down2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)
    
    down3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(pool2)
    down3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(down3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)
    
    down4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(pool3)
    down4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(down4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)
    
    #中间
    center5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(pool4)
    center5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(center5)
    
    #上采样
    up6 = Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'same')(center5)
    up6 = concatenate([up6,down4], axis = 3)
    up6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(up6)
    up6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(up6)

    up7 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(up6)
    up7 = concatenate([up7,down3], axis = 3)
    up7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(up7)
    up7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(up7)

    up8 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(up7)
    up8 = concatenate([up8,down2], axis = 3)
    up8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up8)
    up8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up8)

    up9 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(up8)
    up9 = concatenate([up9,down1], axis = 3)
    up9 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(up9)
    up9 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(up9)
           
    outputs = Conv2D(1, 1, activation = 'sigmoid')(up9)

    model = Model(input = inputs, output = outputs)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
           
model = Unet(input_size = (512,512,1))
model.summary()


from sklearn.model_selection import train_test_split

train_vol, validation_vol, train_seg, validation_seg = train_test_split((X_data-127.0)/127.0, 
                                                            (Y_data>127).astype(np.float32), 
                                                            test_size = 0.1,random_state = 2018)
train_vol, test_vol, train_seg, test_seg = train_test_split(train_vol,train_seg, 
                                                            test_size = 0.1, 
                                                            random_state = 2018)

history = model.fit(x = train_vol,
                    y = train_seg,
                    batch_size = 16,
                    epochs = 20,
                    validation_data = (test_vol,test_seg),
                    verbose = 1)
                    
