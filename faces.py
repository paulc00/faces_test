from utils import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Reshape, Flatten, Dense, Activation
from keras import optimizers                       #import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import losses
from datetime import timedelta, datetime

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))

model_specs = [
'''
[
    Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=X_train.shape[1:]),
    Activation('relu'),
    Flatten(),
    Dense(500),
    Activation('relu'),
    Dense(30),
]
''',
'''
[
    Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=X_train.shape[1:]),
    Activation('relu'),
    Conv2D(filters=64, kernel_size=3, strides=2, padding='same', input_shape=X_train.shape[1:]),
    Activation('relu'),
    Flatten(),
    Dense(500),
    Activation('relu'),
    Dense(500),
    Dense(30),
]
''',
'''
[
    Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=X_train.shape[1:]),
    Activation('relu'),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
    Activation('relu'),
    Flatten(),
    Dense(500),
    Dropout(0.2),
    Activation('relu'),
    Dropout(0.2),
    Dense(500),
    Dense(30),
]
''',
]

model = Sequential(eval(model_specs[0]))
for model_spec in model_specs:
    model = Sequential(eval(model_spec))
# model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
# model.add(MaxPool2D(pool_size=2))
# # model.add(Dropout(0.2))
# # model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
# # model.add(MaxPool2D(pool_size=2))
# # model.add(Dropout(0.2))
# # model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
# # model.add(MaxPool2D(pool_size=2))
# # model.add(Dropout(0.3))
# model.add(Flatten())
# # model.add(Dense(500, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(30))

# Summarize the model
    print(model_specs[0])
    model.summary()


    start_time = datetime.utcnow().timestamp()


    optimizer = optimizers.Adam()
    max_epochs = 50
    batch_size = 128
    validation_split = 0.2

    ## TODO: Compile the model
    model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=['accuracy'])

    # from keras.callbacks import ModelCheckpoint, TensorBoard

    # checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
    #                                save_best_only=True)

    # tensorboard = TensorBoard(log_dir='/mnt/F/tflogs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # train the model
    history = model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size,
                        epochs=max_epochs, verbose=0, shuffle=True)
                        # epochs = max_epochs, callbacks = [checkpointer], verbose = 2, shuffle = True)
    #                  epochs=max_epochs, callbacks=[checkpointer, tensorboard], verbose=2, shuffle=True)

    ## TODO: Save the model as model.h5
    model.save('my_model.h5')

    i = history.history['val_loss'].index(min(history.history['val_loss']))
    print("\nLowest val loss: {}\nVal accuracy: {}\nTrain accuracy: {}".format(history.history['val_loss'][i], history.history['val_acc'][i], history.history['acc'][i]))

    print("\nElapsed time: {}".format(str(timedelta(seconds=round(datetime.utcnow().timestamp() - start_time)))))


    # Plot the loss and accuracy
    fig = plt.figure(figsize = (16,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(history.history['loss'][4:])
    ax1.plot(history.history['val_loss'][4:])
    ax1.set_title('Model Loss')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')

    ax2 = fig.add_subplot(212)
    ax2.plot(history.history['acc'][4:])
    ax2.plot(history.history['val_acc'][4:])
    ax2.set_title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    # model.load_weights('model.weights.best.hdf5')
