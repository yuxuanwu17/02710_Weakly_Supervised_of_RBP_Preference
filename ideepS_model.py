def set_cnn_model(input_dim, input_length):
    nbfilter = 16
    model = Sequential()
    model.add(Convolution1D(input_dim=input_dim, input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    return model


def get_cnn_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    nbfilter = 16
    print
    'configure cnn network'

    seq_model = set_cnn_model(4, 111)
    struct_model = set_cnn_model(6, 111)
    # pdb.set_trace()
    model = Sequential()
    model.add(Merge([seq_model, struct_model], mode='concat', concat_axis=1))

    model.add(Bidirectional(LSTM(2 * nbfilter)))

    model.add(Dropout(0.10))

    model.add(Dense(nbfilter * 2, activation='relu'))
    print
    model.output_shape

    return model


def get_cnn_network_old():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print
    'configure cnn network'
    nbfilter = 16

    model = Sequential()
    model.add(Convolution1D(input_dim=4, input_length=111,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(nbfilter, activation='relu'))

    model.add(Dropout(0.25))

    return model


def get_struct_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print
    'configure cnn network'
    nbfilter = 16

    model = Sequential()
    model.add(Convolution1D(input_dim=6, input_length=111,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(nbfilter, activation='relu'))

    model.add(Dropout(0.25))

    return model
