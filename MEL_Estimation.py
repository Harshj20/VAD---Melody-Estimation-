from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout, Bidirectional, LSTM, concatenate, TimeDistributed, LeakyReLU, Reshape, add, Lambda
from tensorflow.keras.models import Model
from keras.regularizers import l2
import math
from scipy.signal import medfilt
import numpy as np
import librosa
import os
import sys
import argparse

def ResNet_Block(input,block_id,filterNum):
    ''' Create a ResNet block
    Args:
        input: input tensor
        filterNum: number of output filters
    Returns: a keras tensor
    '''
    x = BatchNormalization()(input)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((1, 4))(x)

    init = Conv2D(filterNum, (1, 1), name='conv'+str(block_id)+'_1x1', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_1',padding='same',kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(filterNum, (3, 3),  name='conv'+str(block_id)+'_2',padding='same',kernel_initializer='he_normal',use_bias=False)(x)

    x = add([init, x])
    return x

def create_model(output_size, activation="softmax"):
    num_output = output_size
    input = Input(shape=(31, 513, 1))

    block_1 = Conv2D(64, (3, 3), name='conv1_1', padding='same', kernel_initializer='he_normal', use_bias=False,
                    kernel_regularizer=l2(1e-5))(input)
    block_1 = BatchNormalization()(block_1)
    block_1 = LeakyReLU(0.01)(block_1)
    block_1 = Conv2D(64, (3, 3), name='conv1_2', padding='same', kernel_initializer='he_normal', use_bias=False,
                    kernel_regularizer=l2(1e-5))(block_1)

    block_2 = ResNet_Block(input=block_1, block_id=2, filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3, filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4, filterNum=256)

    block_4 = BatchNormalization()(block_4)
    block_4 = LeakyReLU(0.01)(block_4)
    block_4 = MaxPooling2D((1, 4))(block_4)
    block_4 = Dropout(0.5)(block_4)

    numOutput_P = 2 * block_4.shape[3]
    output = Reshape((31, numOutput_P))(block_4)

    output = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))(output)
    output = TimeDistributed(Dense(num_output))(output)
    output = TimeDistributed(Activation(activation), name='output')(output)

    block_1 = MaxPooling2D((1, 4 ** 4))(block_1)
    block_2 = MaxPooling2D((1, 4 ** 3))(block_2)
    block_3 = MaxPooling2D((1, 4 ** 2))(block_3)

    joint = concatenate([block_1, block_2, block_3, block_4])
    joint = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,
                    kernel_regularizer=l2(1e-5))(joint)
    joint = BatchNormalization()(joint)
    joint = LeakyReLU(0.01)(joint)
    joint = Dropout(0.5)(joint)

    num_V = joint.shape[3] * 2
    output_V = Reshape((31, num_V))(joint)

    output_V = Bidirectional(LSTM(32, return_sequences=True, stateful=False, recurrent_dropout=0.3, dropout=0.3))(
        output_V)
    output_V = TimeDistributed(Dense(2))(output_V)
    output_V = TimeDistributed(Activation("softmax"))(output_V)

    output_NS = Lambda(lambda x: x[:, :, 0])(output)
    output_NS = Reshape((31, 1))(output_NS)
    output_S = Lambda(lambda x: 1 - x[:, :, 0])(output)
    output_S = Reshape((31, 1))(output_S)
    output_VV = concatenate([output_NS, output_S])

    output_V = add([output_V, output_VV])
    output_V = TimeDistributed(Activation("softmax"), name='output_V')(output_V)

    model = Model(inputs=input, outputs=[output, output_V])
    return model



def preprocess_audio(file_name, win_size=31):
    # Load the audio file
    y, sr = librosa.load(file_name, sr=8000, mono=True)
    
    # Compute the Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=80, win_length=1024))
    
    # Convert to decibel scale
    db_S = librosa.amplitude_to_db(S, ref=np.max)
    
    # Normalize between 0 and 1
    norm_db_S = (db_S - np.min(db_S)) / (np.max(db_S) - np.min(db_S))
    
    # Padding for consistent window sizes
    num_frames = norm_db_S.shape[1]
    pad_num = num_frames % win_size
    if pad_num != 0:
        pad_length = win_size - pad_num
        padding_feature = np.zeros(shape=(513, pad_length))
        norm_db_S = np.concatenate((norm_db_S, padding_feature), axis=1)
    
    # Splitting the frames into windows
    x_test = [norm_db_S[:, j:j + win_size].T for j in range(0, norm_db_S.shape[1], win_size)]
    x_test = np.array(x_test)
    x_test = x_test[..., np.newaxis]  # Add a channel dimension for compatibility with CNNs

    return x_test, norm_db_S

def main(file_path, output_dir, model_ = 'c_model_0'):
    pitch_range = np.arange(38, 83 + 1.0/16, 1.0/16)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    X_input, s_db = preprocess_audio(file_path)
    output_size = 722
    Activation = "softmax"
    if(model_[0] == 'r'):
        output_size = 1
        Activation = "relu"
    model = create_model(output_size, Activation)
    model.load_weights('./' + model_)

    y_predict = model.predict(X_input, batch_size=64, verbose=1)

    num_total = y_predict[0].shape[0] * y_predict[0].shape[1]
    est_pitch = np.zeros(num_total)
    y_predict = np.reshape(y_predict[0], (num_total, y_predict[0].shape[2]))  # origin
    
    if(model_[0] == 'c'):
        for i in range(y_predict.shape[0]):
            index_predict = np.argmax(y_predict[i, :])
            pitch_MIDI = pitch_range[np.int32(index_predict)]
            if pitch_MIDI >= 38 and pitch_MIDI <= 83:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440
    elif(model_[0] == 'r'):
        for i in range(y_predict.shape[0]):
            pitch_MIDI = y_predict[i]
            if pitch_MIDI >= 38 and pitch_MIDI <= 83:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440
    else:
        print("please check the model. It should either start with c for classification or r for regression")
        exit(0)

    est_pitch = medfilt(est_pitch, 5)

    PATH_est_pitch = output_dir+'/pitch_'+file_path.split('/')[-1]+'.csv'

    if not os.path.exists(os.path.dirname(PATH_est_pitch)):
        os.makedirs(os.path.dirname(PATH_est_pitch))
    f = open(PATH_est_pitch, 'w')
    for j in range(len(est_pitch)):
        est = "%.2f,%.4f\n" % (0.01 * j, est_pitch[j])
        f.write(est)
    f.close()
    print("file creatied")
def parser():
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--filepath',
                   help='Path to input audio (default: %(default)s',
                   type=str, default='test_audio_file.wav')
    p.add_argument('-o', '--output_dir',
                   help='Path to output folder (default: %(default)s',
                   type=str, default='./')
    p.add_argument('-m', '--model_',
                   help='path to model to be used (default: %(default)s',
                   type=str, default='c_model_0')
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    main(args.filepath, args.output_dir, args.model_)