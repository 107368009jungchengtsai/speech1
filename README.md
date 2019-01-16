# speech-recognition
作法說明


## 1.宣告和定義
    import os
    import numpy as np
    from scipy.fftpack import fft
    from scipy.io import wavfile
    from scipy import signal
    from glob import glob
    import re
    import pandas as pd
    import gc
    from scipy.io import wavfile

    from keras import optimizers, losses, activations, models
    from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
    from sklearn.model_selection import train_test_split
    import keras
    
## 2.train_data讀檔
    L = 16000
    legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

    #src folders


    train_data_path = 'train/audio/'
    print(os.listdir(train_data_path))
    test_data_path = os.path.join('test/test/')
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/1.PNG)
