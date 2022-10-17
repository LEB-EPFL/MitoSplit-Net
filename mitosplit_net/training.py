import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.layers import concatenate, UpSampling2D, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

def create_model(nb_filters=8, firstConvSize=9, nb_input_channels=1, printSummary=False, ):    
    #Hyperparameters
    optimizer_type = Adam(learning_rate=0.5e-3)
    loss = 'binary_crossentropy'
    metrics = [BinaryAccuracy()]
    
    #Network architecture
    input_shape = (None, None, nb_input_channels)
    inputs = Input(shape=input_shape)

    # Encoder
    print('* Start Encoder Section *')

    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(nb_filters, (firstConvSize, firstConvSize), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(down0)

    down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # Down2 and Up2 are not really used in the moment, because they are skipped, as down1_pool
    # is used in the center layer as input and center is used in up1 as input, not up2
    #down2 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1_pool)
    #down2 = BatchNormalization()(down2)
    #down2 = Activation('relu')(down2)
    #down2 = Conv2D(nb_filters*2, (3, 3), padding='same')(down2)
    #down2 = BatchNormalization()(down2)
    #down2 = Activation('relu')(down2)
    #down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # Center
    print('* Start Center Section *')
    center = Conv2D(nb_filters*4, (3, 3), padding='same')(down1_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(nb_filters*4, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)

    # Decoder (with skip connections to the encoder section)
    print('* Start Decoder Section *')
    #up2 = UpSampling2D((2, 2))(center)
    #up2 = concatenate([down2, up2], axis=3)
    #up2 = Conv2D(nb_filters*2, (3, 3), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    #up2 = Activation('relu')(up2)
    #up2 = Conv2D(nb_filters*2, (3, 3), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    #up2 = Activation('relu')(up2)

    up1 = UpSampling2D((2, 2))(center)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)  # was relu also before
    outputs.set_shape([None, None, None, 1])
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    if printSummary:
        print(model.summary())
    return model


def train_model(model, input_data, output_data, batch_size=16):
    # Split dataset into [test] and [train+valid]
    data_set_test_trainvalid_ratio = 0.2
    data_split_state = None   # Random split on each call
    input_train, input_test, output_train, output_test =  train_test_split(input_data, 
                                                                           output_data, 
                                                                           test_size=data_set_test_trainvalid_ratio, 
                                                                           random_state=data_split_state)
    
    validtrain_split_ratio = 0.2  # % of the seen dataset to be put aside for validation, rest is for training
    max_epochs = 20  # maxmimum number of epochs to be iterated
    batch_shuffle= True   # shuffle the training data prior to batching before each epoch
    
    history = model.fit(input_train, output_train,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        validation_split=validtrain_split_ratio,
                        shuffle=batch_shuffle,
                        verbose=2)
  
    #Frames separated for evaluation
    frames_test = np.arange(0, input_data.shape[0], 1)
    frames_test = shuffle(frames_test, random_state=data_split_state)[0:int(input_data.shape[0]*data_set_test_trainvalid_ratio)]
    
    return history.history, frames_test


