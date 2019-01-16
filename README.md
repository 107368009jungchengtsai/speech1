# speech-recognition
作法說明
##流程圖
![image](https://github.com/107368009jungchengtsai/The-Simpsons-Characters-Recognition-Challenge/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)

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
## labels設置
    legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

    #src folders


    train_data_path = 'train/audio/'
    print(os.listdir(train_data_path))
    test_data_path = os.path.join('test/test/')
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/1.PNG)

## 3.train_data分類
    def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

    def log_specgram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,fs=sample_rate,window='hann',nperseg=nperseg,noverlap=noverlap,detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
    
    def list_wavs_fname(dirpath, ext='wav'):
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    
    labels = []
    for fpath in fpaths:
        
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        print(fpath)
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
            
    return labels, fnames
    
    def pad_audio(samples):
        if len(samples) >= L: return samples
        else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

    def chop_audio(samples, L=16000, num=20):
        for i in range(num):
            beg = np.random.randint(0, len(samples) - L)
            yield samples[beg: beg + L]

    def label_transform(labels):
        nlabels = []
        for label in labels:
            if label == '_background_noise_':
                nlabels.append('silence')
            elif label not in legal_labels:
                nlabels.append('unknown')
            else:
                nlabels.append(label)
        return pd.get_dummies(pd.Series(nlabels))
        
## 4.設為16000
    new_sample_rate = 16000
    y_train = []
    x_train = []
    i=0
    dirs = [f for f in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, f))]
    dirs.sort()
    print('Number of labels: ' + str(len(dirs[:])))
    print(dirs)
    for label in dirs[:]:
    waves = [f for f in os.listdir(os.path.join(train_data_path, label)) if f.endswith('.wav')]
    #label_value[label] = i
    i = i + 1
    print(str(i)+":" +str(label) + " ", end="")
    print("\n")
    for fname in waves:
        sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
        samples = pad_audio(samples)
        if len(samples) > 16000:
            n_samples = chop_audio(samples)
        else: n_samples = [samples]
        for samples in n_samples:
            resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
            _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)

    x_train = np.array(x_train)
    x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
    y_train = label_transform(y_train)
    print(y_train.columns.values)
    label_index = y_train.columns.values
    y_train = y_train.values
    y_train = np.array(y_train)
    print (x_train.shape)
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/2.PNG)
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/3.PNG)
## 5.權重
    from keras.models import Sequential, Model, load_model
    from keras.layers import merge, Input, Dense, TimeDistributed, Lambda
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
    from keras.callbacks import ReduceLROnPlateau
    input_shape = (99, 161, 1)
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(filters=8, kernel_size=2, activation='relu'))
    model.add(Convolution2D(filters=8, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(filters=16, kernel_size=3, activation='relu'))
    model.add(Convolution2D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(12,activation='softmax'))
## 6.查看model
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/4.PNG)
## 7.訓練次數
    epochs = 30
    batch_size = 16
    file_name = str(epochs) + '_' + str(batch_size)
## 8.split train_data/valid_data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=0.01,mode='min')]
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=
    (x_test,y_test),callbacks=callbacks)
    #model.save('h5/' + file_name + '.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/5.PNG)
## 9.mode save
    model.save('model.h5')
## 10.test_data讀檔設16000
    new_sample_rate = 16000
    def test_data_generator(batch=16):
        fpaths = glob(os.path.join(test_data_path, '*wav'))
        i = 0
        for path in fpaths:
            print(path)
            if i == 0:
                imgs = []
                fnames = []
            i += 1
            rate, samples = wavfile.read(path)
            samples = pad_audio(samples)
            resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
            _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
            imgs.append(specgram)
            fnames.append(path.split('/')[-1])
            #print(fnames)
            if i == batch:
                i = 0
                imgs = np.array(imgs)
                imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
                yield fnames, imgs
        if i < batch:
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
        raise StopIteration()
        from keras.models import Sequential, Model, load_model
## 11.labels 順序
    test_data=[]
    test_data_name = []
    index = []
    results = []
    label_index = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop', 'unknown', 'up', 'yes']
## 12.共有10500筆
    for i in range(10500):
        sample_rate, samples = wavfile.read(os.path.join(test_data_path+str(i+1)+'.wav'))
        samples = pad_audio(samples)
        if len(samples) > 16000:
            n_samples = chop_audio(samples)
        else: n_samples = [samples]
        for samples in n_samples:
            resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
            _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        test_data_name.append(i+1)
        test_data.append(specgram)
    test_data = np.array(test_data)
    test_data=test_data.reshape(tuple(list(test_data.shape) + [1]))
    model = load_model('model.h5')
    predict = model.predict(test_data)
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(10500):
        index.append(i+1)
        print(label_index[predict[i]])
    results.append(label_index[predict[i]])
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/6.PNG)

## 13.測資存進csv
    df = pd.DataFrame(columns=['id', 'word'])
    df['id'] = index
    df['word'] = results
    df.to_csv(os.path.join('sub.csv'), index=False)
    
## 14.kaggle排名
![image](https://github.com/107368009jungchengtsai/speech1/blob/master/7.PNG)
## 15.分析
    這一次參考各種殼層寫法，發覺現在使用的這種表現較佳，做資料分析時，有用到BatchNormalization能有效的提高準確度，但這次提高epochs對於準確度卻是沒
    有什麼幫助。
## 16.改進
    這次使用的mode是從TensorFlow Speech Recognition Challenge參考上來的，準確率有達到9成以上，但往後無法再提升，可能可以加雜訊，讓model判斷更加
    完善。
