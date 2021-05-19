# data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import pandas as pd
import re

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

class cnn_impressionists:
    
    def __init__(self,args):
        self.args = args
        
    def preprocess(self):
        #Path to training data
        train_path = self.args["training_path"]

        #Listing artists
        artists = os.listdir(train_path)
        artists.pop(4)

        #Define empty lists to store image paths and Y values for training data
        self.all_img=[]
        self.trainY = []

        #Loop through artists and extract filenames
        for i in artists:
            imgs = os.listdir(os.path.join(train_path, i))

            #Add artist names and image paths to lists
            self.trainY.extend(np.repeat(i, len(os.listdir(os.path.join(train_path, i)))  ))
            self.all_img = self.all_img +["/".join([train_path, i, n]) for n in imgs]


        #same process for testing data
        test_path = self.args["validation_path"]

        #Empty lists for storing testing image paths and Y values
        self.test_img=[]
        self.testY = []

        #looping through each artist in the validation folder
        for i in artists:
            imgs = os.listdir(os.path.join(test_path, i))
            self.testY.extend(np.repeat(i, len(os.listdir(os.path.join(test_path, i)))  ))
            self.test_img = self.test_img +["/".join([test_path, i, n]) for n in imgs]

        #Defining dimensions to resize the images
        Height, Width = re.split('H|W',self.args["image_size"])[1:]
        dim = (int(Height),int(Width))

        #Loading each image, resizing them and saving in list
        self.trainX = [cv2.resize(cv2.imread(filepath), dim, interpolation = cv2.INTER_AREA) for filepath in self.all_img]
        self.testX = [cv2.resize(cv2.imread(filepath), dim, interpolation = cv2.INTER_AREA) for filepath in self.test_img]

        #Normalize images
        self.trainX = np.array(self.trainX).astype('float')/255.
        self.testX = np.array(self.testX).astype('float')/255.

        # converting artist names to one-hot vectors
        lb = LabelBinarizer()
        self.trainY = lb.fit_transform(self.trainY)
        self.testY = lb.fit_transform(self.testY)

        # initialize label names
        self.labelNames = artists
    
    def design_model(self):
         # define model
        self.model = Sequential()

        # first set of CONV => RELU => POOL
        self.model.add(Conv2D(self.args["hidden_layers"], (3, 3), 
                         padding="same", 
                         input_shape=(200, 200, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2)))

        # second set of CONV => RELU => POOL
        self.model.add(Conv2D(50, (5, 5), 
                         padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2)))

        # FC => RELU
        self.model.add(Flatten())
        self.model.add(Dense(500))
        self.model.add(Activation("relu"))

        # softmax classifier
        self.model.add(Dense(10))
        self.model.add(Activation("softmax"))
        
        #define learning rate and compile model
        opt = SGD(lr=self.args["learning_rate"])
        self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    def fit_model(self):
        
        self.H = self.model.fit(self.trainX, self.trainY, 
          validation_data=(self.testX, self.testY), 
          batch_size=32,
          epochs=self.args["epochs"],
          verbose=1)

        #Predicting the test set
        self.predictions = self.model.predict(self.testX, batch_size=32)

        #Saving classification report 
        results_df = pd.DataFrame(classification_report(self.testY.argmax(axis=1),
                                    self.predictions.argmax(axis=1),
                                    target_names=self.labelNames,
                                    output_dict=True)).transpose()
        
        if self.args["output"] is not None:
            output_path = os.path.join(self.args["output"],
                                       "classification_report.csv")
        else:
            output_path = "classification_report.csv"

        results_df.to_csv(output_path)
        
        
    
    #Define plot_history function
    def plot_history(self):
        # visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.args["epochs"]),
                 self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.args["epochs"]),
                 self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.args["epochs"]),
                 self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.args["epochs"]),
                 self.H.history["val_accuracy"], label="val_acc")
        
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()


        
        if self.args["output"] is not None:
            output_path = os.path.join(self.args["output"],
                                       "history.png")
        else:
            output_path = "history.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    #Define main function
def main():
    #Add the terminal argument
    ap = argparse.ArgumentParser()

    #Let users define # of epochs
    ap.add_argument("-e","--epochs", default = 10,type = int,
                    help="Specify amount of epochs, default: 10" )
    ap.add_argument("-t","--training_path", required=True,type = str,
                    help="Specify training path" )
    ap.add_argument("-v","--validation_path", required=True,type = str,
                    help="Specify validation path" )
    ap.add_argument("-s","--image-size", required=False,
                    default = "H200W200", type = str,
                    help="Specify desired image size. \
                    larger images will slow down computation")
    ap.add_argument("-l","--learning-rate", required=False,
                    default = 0.01, type = float,
                    help="Specify learning rate")
    ap.add_argument("--hidden-layers", required=False,
                    default = 32, type = int,
                    help="Specify amount of hidden layers")
    ap.add_argument("-o","--output", required=False,
                    default = None, type = str,
                    help="Specify output path")

    #parse arguments
    args = vars(ap.parse_args())

    cnn_artists = cnn_impressionists(args)
    print("loading and preprocessing images...")
    cnn_artists.preprocess()
    cnn_artists.design_model()
    print("Fitting model...")
    cnn_artists.fit_model()
    cnn_artists.plot_history()

if __name__=="__main__":
    main()
