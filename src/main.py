import keras.utils
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.datasets import cifar10
from keras.layers import Flatten, Dropout
from keras.layers import Input, Conv2D, Dense, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

num_classes = 10
batch_size = 64
DATA_FORMAT = 'channels_last'
log_filepath = './process'
factor = 10


def data_preprocess(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    line = 3
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.0, 66.7]
    for i in range(line):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(number):
    if number < 100:
        return 0.01
    if number < 200:
        return 0.001
    return 0.0001


def Alexnet(input, DROPOUT, number=10, less_layers=False):
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
               activation='relu', kernel_initializer='uniform')(input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    if not less_layers:
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='uniform')(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='uniform')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(number, activation='softmax')(x)
    return out


def Alexnet_sigmoid(input, DROPOUT, number=10, less_layers=False):
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
               activation='sigmoid', kernel_initializer='uniform')(input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='sigmoid', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='sigmoid', kernel_initializer='uniform')(x)
    if not less_layers:
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                   activation='sigmoid', kernel_initializer='uniform')(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                   activation='sigmoid', kernel_initializer='uniform')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(4096, activation='sigmoid')(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(number, activation='softmax')(x)
    return out

def Alexnet_tanh(input, DROPOUT, number=10, less_layers=False):
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
               activation='tanh', kernel_initializer='uniform')(input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='tanh', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='tanh', kernel_initializer='uniform')(x)
    if not less_layers:
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                   activation='tanh', kernel_initializer='uniform')(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                   activation='tanh', kernel_initializer='uniform')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(number, activation='softmax')(x)
    return out


def experiment1():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[::factor]
    y_train = y_train[::factor]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = data_preprocess(x_train, x_test)
    DROPOUT = 0.5
    input = Input(shape=(32, 32, 3))
    output = Alexnet(input, DROPOUT)
    model = Model(input, output)
    model.summary()
    weight_decay = 0.0005
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)
    epochs = 100
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=700 / factor,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test))

    model.save('alexnet.e1')


def experiment2():
    """
    Less dropout
    :return:
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[::factor]
    y_train = y_train[::factor]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = data_preprocess(x_train, x_test)
    DROPOUT = 0.1
    input = Input(shape=(32, 32, 3))
    output = Alexnet(input, DROPOUT)
    model = Model(input, output)
    model.summary()
    weight_decay = 0.0005
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)
    epochs = 100
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=700 / factor,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test))

    model.save('alexnet.e2')


def experiment3():
    """
    sigmoid
    :return:
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[::factor]
    y_train = y_train[::factor]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = data_preprocess(x_train, x_test)
    DROPOUT = 0.5
    input = Input(shape=(32, 32, 3))
    output = Alexnet_sigmoid(input, DROPOUT)
    model = Model(input, output)
    model.summary()
    weight_decay = 0.0005
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)
    epochs = 100
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=700 / factor,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test))

    model.save('alexnet.e3')
    
    
def experiment_tanh():
    """
    tanh
    :return:
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[::factor]
    y_train = y_train[::factor]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = data_preprocess(x_train, x_test)
    DROPOUT = 0.5
    input = Input(shape=(32, 32, 3))
    output = Alexnet_tanh(input, DROPOUT)
    model = Model(input, output)
    model.summary()
    weight_decay = 0.0005
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)
    epochs = 100
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=700 / factor,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test))

    model.save('alexnet.etanh')


def experiment4():
    """
    less layers
    :return:
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[::factor]
    y_train = y_train[::factor]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = data_preprocess(x_train, x_test)
    DROPOUT = 0.5
    input = Input(shape=(32, 32, 3))
    output = Alexnet(input, DROPOUT, less_layers=True)
    model = Model(input, output)
    model.summary()
    weight_decay = 0.0005
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',
                                 cval=0.)

    datagen.fit(x_train)
    epochs = 100
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=700 / factor,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test))

    model.save('alexnet.e4')


#experiment1()
#experiment2()
#experiment3()
experiment4()
#experiment_tanh()
