#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[6]:


with open("data_batch_1 (1)","rb") as f:
    batch=pickle.load(f,encoding='bytes')


# In[7]:


X=batch[b'data']
y=np.array(batch[b'labels'])


# In[8]:


X_images=X.reshape(-1,3,32,32).transpose(0,2,3,1).astype("float32")/255.0


# In[10]:


y_cat=to_categorical(y,num_classes=10)


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(X_images,y_cat,test_size=0.2,random_state=42)
print("data loaded:",X_train.shape,X_test.shape)


# In[18]:


model_lenet = models.Sequential([
    layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3),padding='same'),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, (5, 5), activation='tanh'),
    layers.AveragePooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])


# In[20]:


model_lenet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_lenet.summary()
model_lenet.fit(X_train,y_train,epochs=5,batch_size=64,
                validation_data=(X_test,y_test))


# In[ ]:


model_alex=models.Sequential([
    layers.Conv2D(96, (3, 3), activation='relu', input_shape=(32, 32, 3),padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(384, (3, 3), activation='relu',padding='same'),
    layers.Conv2D(384, (3, 3), activation='relu',padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='tanh'),
    layers.Dense(512, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

model_alex.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_alex.summary()
model_alex.fit(X_train,y_train,epochs=5,batch_size=64,
                validation_data=(X_test,y_test))



# In[ ]:




