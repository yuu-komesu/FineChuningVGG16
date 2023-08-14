'''
Create a model with command line interface.

Usage:
python3 fine_chuning_vgg16.py --train-directory-path /path/to/train_dir --test-directory-path /path/to/test_dir

'''
import tensorflow as tf
import numpy as np
import os
import cv2
import imghdr
import argparse
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# Remove dodgy images
def path_rm_img(data_dir):
    """Removes images from the specified directory that do not have valid extensions.
    Args:
        data_dir (str): The directory path containing the image files.
    Returns:
        None
    Raises:
        None
    """
    n=0
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            if n==0:
                print(image_path)
                n+=1
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                os.remove(image_path)

                
def split_files(dir_path):

    print(dir_path)
    if os.path.exists(dir_path):
        path_rm_img(dir_path)
    else:
        print("The specified directory does not exist.")
    return dir_path


def train_data_split(train_dir):
    
    train_dir_path = split_files(train_dir)
    # Load train data
    train_data = tf.keras.utils.image_dataset_from_directory(train_dir_path)
    train_data = train_data.map(lambda x,y: (x/255, y)) # Scaling data


    # split data
    train_dataset_size=int(len(train_data))
    train_size=int(train_dataset_size*.7)
    val_size=int(train_dataset_size*.2)
    test_size=int(train_dataset_size-train_size-val_size)

    print("train dataset size: {}\ntrain size: {}\nval size: {}\ntest size:{}".format(train_dataset_size,train_size,val_size,test_size))

    train = train_data.take(train_size)
    val = train_data.skip(train_size).take(val_size)
    test = train_data.skip(train_size+val_size).take(test_size)
    
    return train, val, test

def test_data_load(test_dir):

    test_dir_path = split_files(test_dir)
    test_data=tf.keras.utils.image_dataset_from_directory(test_dir_path)
    test_data=test_data.map(lambda x,y: (x/255, y)) # Scaling data
    
    return test_data

def model_build():

    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load VGG16 model
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False ## Not trainable weights

    # Build fine chuning model
    model = Sequential()

    model.add(base_model)

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu")) 
    model.add(Dense(64, activation="relu")) 
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 

    model.compile(
        optimizer= 'adam', 
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model


def fine_chuning(model, train, val):

    
    # model train

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    print(hist.history)

    # Plot performance

    fig = plt.figure() # plot loss
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("train_loss.png")


    fig = plt.figure() # plot accracy
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("train_accracy.png")
    
    return model

    
def evaluate(model, test):
    # evaluate by train test dataset

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    test_result="Precision result: {}\nRecall result: {}\nBinary Accuracy result: {}".format(pre.result(), re.result(), acc.result())
    
    return test_result

    
def evaluate_by_dataset(model, test, test_data):
    
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    
    train_test_result = evaluate(model, test)
    f = open('train_test_result.txt', 'w')
    f.write(train_test_result)
    f.close()

    pre.reset_state()
    re.reset_state()
    acc.reset_state()
    
    # evaluate by test dataset

    test_result=evaluate(model, test_data)
    f = open('test_result.txt', 'w')
    f.write(test_result)
    f.close()

def model_save(model):
    # Save the model

    model.save(os.path.join('models','imageclassifier_VGG16.h5'))
    
    
def main(args):
    train_path = args.train_directory_path
    test_path = args.test_directory_path
    model = model_build()
    train, val, test = train_data_split(train_path)
    test_data = test_data_load(test_path)
    model = fine_chuning(model, train, val)
    evaluate_by_dataset(model, test, test_data)
    model_save(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-directory-path", type=str, required=True)
    parser.add_argument("--test-directory-path", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
