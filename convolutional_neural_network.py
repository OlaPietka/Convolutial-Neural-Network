import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Unormowanie danych
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# Zamiana liczb na ciągi 10 znaków
y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)


# Dodajemy wymiar jako rozmiar danych treningowych/testowych
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # Liczba danych, rozmiar pixeli, rozmiar pixeli, wymiar kolorow
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) # Liczba danych, rozmiar pixeli, rozmiar pixeli, wymiar kolorow

model = keras.models.Sequential()

# Warstwa 1
model.add(keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                              input_shape=(28, 28, 1), padding='same'))
# Filtrow 6, rozmiar filtra, krok, funkcja aktywacji, rozmiar wejsciowu, padding


# Warstwa 2
model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Usrednienie, krok, padding


# Warstwa 3
model.add(keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))


# Warstwa 4
model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


# Warstwa 5
model.add(keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))


# Warstwa 6
model.add(keras.layers.Flatten()) # Splaszczenie
model.add(keras.layers.Dense(84, activation='tanh'))


# Warstwa 7
model.add(keras.layers.Dense(10, activation='softmax'))


# Kompilacja modelu
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
# Funckja bledu, optymizer, metryka (np accuarcy) liczenia ile probek zostalo rozpoznanych (klucz tablicy)


# Uczenie
EPOCHS = 10
History = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=128, validation_split=0.01, verbose=1)
# batch= ile na raz probek dajemy na siec
# validationsplit = uczenie 60tys - 20%
# verbose = tryb rozmowny, pojawi sie wiecej informacji niz zwykle


# Nagranie na dysk
model.save('mnist.model')


# Testowanie
test_score = model.evaluate(x_test, y_test)
print("Loss: {:.5f}, accuracy: {:.3}%".format(test_score[0], test_score[1]*100))


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(EPOCHS), History.history["loss"], label="train loss")  # os x, os y
plt.plot(range(EPOCHS), History.history["val_loss"], label="validation loss")
plt.plot(range(EPOCHS), History.history["accuracy"], label="train accuracy")
plt.plot(range(EPOCHS), History.history["val_accuracy"], label="validation accuracy")
plt.savefig('mnist.png')