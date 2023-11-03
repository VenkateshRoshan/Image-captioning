import numpy as np
import pandas as pd
import re
import os

from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import keras
from keras_preprocessing import image
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model , Model
from keras.layers import Dense , Dropout , LSTM , Embedding , Input , add , Conv2D , MaxPooling2D , Flatten , Bidirectional , BatchNormalization
from keras import preprocessing , applications

from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv('descriptions.csv',delimiter=',', encoding='cp1252')

descriptions = df.description
train_data, test_data = train_test_split(df, test_size=0.1)

Images_path = 'images/images/'

nltk.download("stopwords")
nltk.download("punkt")

def cleaning(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        #stemming
        words.append([i.lower() for i in w])
    return words

cleaned_words = cleaning(descriptions)
Train_cleaned_words = cleaning(train_data.description)
Test_cleaned_words = cleaning(test_data.description)
print(cleaned_words[:5])

# Adding 'StSeq' and 'EndSeq' to the cleaned_words list
for cleaned_desc in cleaned_words :
    cleaned_desc.insert(0,'stseq')
    cleaned_desc.insert(len(cleaned_desc),'endseq')

for cleaned_desc in Train_cleaned_words :
    cleaned_desc.insert(0,'stseq')
    cleaned_desc.insert(len(cleaned_desc),'endseq')
    
for cleaned_desc in Test_cleaned_words :
    cleaned_desc.insert(0,'stseq')
    cleaned_desc.insert(len(cleaned_desc),'endseq')

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

def max_length(words):
    return(len(max(words, key = len)))

def encoding_doc(token, words):
    return token.texts_to_sequences(words)

word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)
print(f"Vocab Size = {vocab_size} and Maximum length = {max_length}")

word_dictionary = dict(word_tokenizer.word_index)
#print(word_dictionary)
print(word_dictionary['stseq'] , word_dictionary['endseq'])

dict_to_word = {}

for key , val in zip(word_dictionary.keys() , word_dictionary.values()) :
    dict_to_word[str(val)] = key



Train_Images = []
Test_Images = []
img_shape = (64,64)

def Load_Images(img_path,data_frame) :
    Images = []
    for file_loc in data_frame.file :
        img = cv2.imread(img_path+file_loc)
        img = cv2.resize(img , img_shape)
        Images.append(img)
    Images = np.array(Images)/255.
    return Images
    
Train_Images = Load_Images(Images_path,train_data)
Test_Images = Load_Images(Images_path,test_data)

print(Train_Images.shape , Test_Images.shape)

Train_cleanwords = []
Test_cleanwords = []
Train_padded_doc = []
Test_padded_doc = []

X_train_input_1 = []
X_train_input_2 = []
Y_train = []

for index,desc in enumerate(Train_cleaned_words) :
    k = len(desc)
    #print(desc)
    for i in range(1,k) :
        partial_desc = []
        w = 0
        for w in range(0,i) :
            partial_desc.append(str(desc[w]).lower())
        X_train_input_1.append(Train_Images[index])
        X_train_input_2.append(partial_desc)
        Y_train.append(desc[w+1])

def padding_doc(encoded_doc, max_length):
    return pad_sequences(encoded_doc, maxlen = max_length, padding = "post")

encoded_doc = encoding_doc(word_tokenizer , X_train_input_2)
Train_padded_doc = padding_doc(encoded_doc, max_length)
print("Shape of Trained padded docs = ",Train_padded_doc.shape)

X_train_input_1 = np.array(X_train_input_1)
Train_padded_doc = np.array(Train_padded_doc)
Y_train = np.array(Y_train)

### Model

inputs1 = Input(shape=(64,64,3))
fe1 = Conv2D(64,3,activation='relu')(inputs1)
fe2 = MaxPooling2D(2)(fe1)
# fe3 = Conv2D(32,3,activation='relu')(fe2)
# fe4 = MaxPooling2D(2)(fe3)
fe5 = Flatten()(fe2)
fe6 = Dense(256, activation='relu')(fe5)
fe7 = Dropout(0.5)(fe6)

# partial caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 200 , mask_zero=True)(inputs2)
se2 = Bidirectional(LSTM(128))(se1)
se3 = Dropout(0.5)(se2)

# decoder (feed-forward) model
decoder1 = add([fe7, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
decoder2 = BatchNormalization(synchronized=True)(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.summary()

opt = keras.optimizers.Adam(learning_rate=1e-03)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model_name = 'Task_conv_2.h5'
callbacks   = [
      EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
      ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min')
]

history = model.fit([X_train_input_1,Train_padded_doc], Y_train , epochs=20 , validation_split=.2 , batch_size=32 ,callbacks = callbacks)

for img in Test_Images[:5] :
    seq = ['stseq']
    X_test = []
    X_test.append(img)
    X_test = np.array(X_test)
    print(X_test.shape)
    plt.imshow(X_test[0])
    plt.show()
    for i in range(max_length) :
        desc = [[word_dictionary[i] for i in seq]]
        desc = padding_doc(desc , max_length)
        pred = model.predict([X_test,desc])
        pred = np.argmax(pred)
        word = dict_to_word[str(pred)]
        seq.append(word)
        if word == 'endseq' :
            break
    print(seq)
    image_desc = ""
    for i in seq[1:-1] :
        image_desc += i + " "
    print(image_desc)

