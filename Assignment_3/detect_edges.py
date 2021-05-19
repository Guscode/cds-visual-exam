#!/usr/bin/python

#import packages
import os
from pathlib import Path
import cv2
import argparse
import pandas as pd
import re
import pytesseract


#Create class 
class text_detection:

    #Create init function loading data and creating self.args with arguments from argparse
    def __init__(self, args):
        self.args = args
        self.img = cv2.imread(self.args["image"]) #read image
    
    #Create function for cropping images with coordinates from argparse
    def crop_image(self):
        
        X1, X2, Y1, Y2 = re.split('X|Y',self.args["crop_coordinates"])[1:] #Unpacking coordinate string
        
        (center_X, center_Y) = (self.img.shape[1]//2, self.img.shape[0]//2) #Creating center coordinates
        
        self.img = self.img[center_Y-int(Y1):center_Y+int(Y2), center_X-int(X1):center_X+int(X2)] #Cropping image
        
        
    
    #Create function for text detection and text extraction
    def extract_text(self):
        
        grey_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) #Converting image to greyscale
                
        blurred = cv2.GaussianBlur(grey_image, (3,3),0) #blurring image to reduce noise
        
        ret,threshold = cv2.threshold(blurred,100,255,cv2.THRESH_BINARY_INV) #using threshold to sharpen image
        
        canny = cv2.Canny(threshold,150,170) #Using canny edge detection to find edges

        (contours, _) = cv2.findContours(canny.copy(), #Defining contours based on canny edges
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        
        self.contoured_image = cv2.drawContours(self.img.copy(), #Applying contours to original image
                        contours,
                        -1,
                        (0,255,0),
                        2)
        
        if self.args["psm"] is not None: #Creating config string if --psm is specified
            py_configs = "--psm "+self.args["psm"]
        else:
            py_configs = "" #if not, use default configs

        

        self.output_text = pytesseract.image_to_string(self.contoured_image, config = py_configs) #Extracting output text\
                                                                                               #with tesseract
    
    #create function for saving multiple contour images
    def save_output_multi(self):
        
        for n, image in enumerate(self.contoured_images): #looping through images
            output_path = os.path.join(self.args["output"],
                                       "_".join([self.image_names[n], "contour.jpg"])) #creating output path
            cv2.imwrite(output_path, image) #writing the image
                        
        df = pd.DataFrame(list(zip(self.image_names, self.all_texts)), #Create dataframe with texts and image names
                   columns=['image_names', 'texts'])
        
        
        df.to_csv(os.path.join(self.args["output"], "texts.csv")) #save dataframe
    
    #create function for saving output from single image
    def save_output(self):
        img_name = "_".join([Path(self.args["image"]).stem, "contour"])
        output_path = os.path.join(self.args["output"], "".join([img_name, ".jpg"])) #creating output path
        cv2.imwrite(output_path, self.contoured_image) #Writing the image
        
        text_output = os.path.join(self.args["output"], "ocr_text.txt") #creating output path for text
        with open(text_output, "w") as text_file: #creating .txt file and printing the text
            print(f"Text in image: {self.output_text}", file=text_file)


    #Create run function
    def run(self):
        if self.args["image_files"] is not None: #If image-files is specified
            
            extensions = ['.jpg', '.png', '.jpeg'] #create list of accepted extensions
            
            #create list with image paths
            self.image_paths = [x for x in Path(self.args["image_files"]).iterdir() if x.suffix.lower() in extensions] 
            self.image_names = [Path(i).stem for i in self.image_paths] #create list with image names                          
            self.contoured_images = [] #create empty list for storing images
            self.all_texts = [] #create empty list for storing texts
            for n, image in enumerate(self.image_paths): #loop through image paths
                self.img = cv2.imread(str(image)) #Read image
                if self.args["crop_coordinates"] is not None: #if crop coordinates are specified, crop the image
                    self.crop_image()
                    
                self.extract_text() #Extract text from image
                self.all_texts.append(self.output_text) #Add text to list
                self.contoured_images.append(self.contoured_image) #add contoured image to list
                                     
            self.save_output_multi() #save images and texts
            
        else: #if running a single image
            if self.args["crop_coordinates"] is not None: #crop if coordinates are specified
                self.crop_image()
            self.extract_text() #extract text and contour image
            self.save_output()


def main():

    #Add all the terminal arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--image", required = False, default = "",
                    help="Path to an image with text" )
    ap.add_argument("--output", required = False,
                    default = "",
                    help="Add output path to store results in a different folder")
    ap.add_argument("--crop-coordinates", required = False,
                    default = None,
                    help="Add coordinates for cropping imput image. format: X1X2Y1Y2 See github for examples")
    ap.add_argument("--image-files", required = False,
                    default = None, type = str,
                    help="Add path to image files if you want to extract multiple images")
    ap.add_argument("--psm", required = False,
                    default = None, type = str,
                    help="Add custom page segmentation mode for pytesseract. For details, visit: https://ai-facets.org/tesseract-ocr-best-practices/")
    
    
    #parse arguments
    args = vars(ap.parse_args())

    #Run everything
    initiate = text_detection(args)
    initiate.run()
    
if __name__=="__main__":
    main()