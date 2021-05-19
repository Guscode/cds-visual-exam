# Assignment 3: Edge Detection

To run this code, please follow the guide for activating the virtual environment in [cds-visual-exam](https://github.com/Guscode/cds-visual-exam).

To test the script, in the virtual environment, please run:
```bash
python detect_edges.py --image jefferson.jpg
```
This will return a contoured version of the original image as well as a text file generated with pytesseract.
For better results, you can include cropping coordinates to crop the image as closely to the text as possible, in order to reduce noise:
```bash
python detect_edges.py --image jefferson.jpg --crop-coordinates X750X700Y750Y1150
```

The results without cropping and the results with cropping:
<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_3/results/jeffersons.png" alt="Logo" width="1100" height="900">
</a>

The script also includes the possibility to perform edge-detection on multiple images.
```bash
python detect_edges.py --image-files signs
```
The results:

<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_3/results/city_signs.png" alt="Logo" width="900" height="900">
</a>

The user defined arguments are:

```bash
--image #Path to an image
--output #Path where you want the output files
--crop-coordinates # X and Y coordinates for cropping image in X1X2Y1Y2 format.
--image-files #Path to a folder with images. script takes all files from folder with .jpg, .jpeg or .png
--psm #Specifies page segmentation method. see below for specifications.

```


