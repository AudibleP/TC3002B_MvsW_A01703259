import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = 'C:\\Users\\ricar\\OneDrive\\Escritorio'
train_dir = os.path.join(base_dir,'train')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
							rescale = 1./255, #obligatorio
							#rotation_range = 40,
							#width_shift_range = 0.2,
							#height_shift_range = 0.2,
							#shear_range = 0.3,
							#zoom_range = 0.3,
							#horizontal_flip = True
							)

train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (150, 150),
							batch_size = 8,
							class_mode ='categorical'
							)




from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape = (150,150,3), padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding='same'))
model.add(layers.MaxPooling2D())
#model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
#model.add(layers.MaxPooling2D())
#model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
#model.add(layers.MaxPooling2D())
#model.add(layers.Conv2D(256, (3, 3), activation="relu", padding='same'))
#model.add(layers.MaxPooling2D())
#model.add(layers.Conv2D(512, (3, 3), activation="relu", padding='same'))
#model.add(layers.MaxPooling2D())
#model.add(layers.Conv2D(1028, (3, 3), activation="relu", padding='same'))
model.add(layers.MaxPooling2D())
#model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
#model.add(layers.Dense(256,activation='relu'))
#model.add(layers.Dropout(rate = 0.1))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Dense(3,activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['acc'])


history = model.fit(
						train_generator,
						epochs = 15)


acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs,acc,'bo',label='train accuracy')
plt.title('train acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss, 'bo', label ='training loss')
plt.title('train loss')
plt.legend()

plt.show()

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
					test_dir,
					target_size = (150, 150),
					batch_size = 1,
					class_mode= 'categorical')

test_loss_original, test_acc_original = model.evaluate(test_generator)
print('\ntest acc :\n', test_acc_original)

plt.figure()
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1, 2, figsize=(10, 3))
axarr[0].plot(epochs,acc,label='train accuracy')
axarr[0].legend()
axarr[1].plot(epochs,loss,label='train loss')
axarr[1].legend()


print(acc)
print(loss)

test_loss, test_acc = model.evaluate(test_generator)
print('\ntest acc :\n', test_acc)
print('\ntest loss :\n', test_loss)

predict = model.predict(test_generator)
predict_class = (predict > 0.5).astype("int32")
predict_class.shape

test_loss, test_acc = model.evaluate(test_generator)
print('\ntest acc :\n', test_acc)
print('\ntest loss :\n', test_loss)

model.save('C:\\Users\\ricar\\OneDrive\\Escritorio\\ModeloGuardadoCategorical.h5')