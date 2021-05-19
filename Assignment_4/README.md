# Assignment 4:


To run this code, please follow the guide for activating the virtual environment in [cds-visual-exam](https://github.com/Guscode/cds-visual-exam).

These scripts will test the image classification performance of a logistic regression and a neural network on the [mnist](https://ieeexplore.ieee.org/abstract/document/6296535?casa_token=-GU8K4SuW7sAAAAA:7YB1ZytBmMwkvA3OWTY6vLe1RRqzDr_mFsSQ0QNCBURmDDBnSD4yaafJErVpXXk0G3WpRkZF8sk) handwritten digits data

To test the logistic regression, in the virtual environment, please run:
```bash
python lr-mnist.py --mnist download --output outputs 
```

To test the logistic regression, in the virtual environment, please run:
```bash
python nn-mnist.py --mnist download --output outputs 
```

The output will be a classification report and a confusion matrix heatmap - this shows the results from lr-mnist.py.
<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_4/outputs/heatmap.png" alt="Logo" width="450" height="450">
</a>


This will return a contoured version of the original image as well as a text file generated with pytesseract.
For better results, you can include cropping coordinates to crop the image as closely to the text as possible, in order to reduce noise:
```bash
python detect_edges.py --image jefferson.jpg --crop-coordinates X750X700Y750Y1150
```
