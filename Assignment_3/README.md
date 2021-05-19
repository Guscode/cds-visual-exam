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

The results without cropping                           The results with cropping:
<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_3/results/jeffersons.png" alt="Logo" width="900" height="900">
</a>

