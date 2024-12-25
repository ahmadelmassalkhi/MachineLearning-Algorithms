from model import model
import lib
import lib.Layers


''' INIT & COMPILE MODEL '''
_model = model(lib.Layers.Flatten(),
               lib.Layers.Dense(128, activation='relu', regularizer='l2'),
               lib.Layers.Dropout(0.5),
               lib.Layers.Dense(10, activation='softmax'))
_model.compile(optimizer='adam', loss='categorical_crossentropy')


''' LOAD DATASET '''
from tensorflow.keras.datasets import mnist # type: ignore
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0 # normalize for faster computation & to prevent exploding outputs
y_train = y_train


''' TRAIN MODEL '''
loss, accuracy = _model.fit(x_train, y_train, epochs=5)
print(f'Training: loss = {loss}, accuracy = {accuracy}')


''' EVALUATE MODEL '''
loss, accuracy = _model.evaluate(x_test, y_test)
print(f'Evaluation: loss = {loss}, accuracy = {accuracy}')