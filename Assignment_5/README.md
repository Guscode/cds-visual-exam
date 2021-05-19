# Assignment 5:

To run this code, please follow the guide for activating the virtual environment in [cds-visual-exam](https://github.com/Guscode/cds-visual-exam).

The script cnn-artists.py will train a convolutional neural network a dataset with impressionist painters found [here](https://www.kaggle.com/delayedkarma/impressionist-classifier-data)

In order to test the script, please download the dataset and place it in the Assignment_5 folder
Then run, in the virtual environment:

```bash
python cnn-artists.py --training-path impressionist/training --validation-path impressionist/validation --output output
```

This will return a classification report in .csv format and a history plot showing the training accuracy, validation accuracy, training loss and validation loss.

<a href="https://github.com/Guscode/cds-visual-exam-2021">
    <img src="/Assignment_5/output/history_plot.png" alt="Logo" width="460" height="340">
</a>


# User Defined Parameters
The user defined arguments for cnn-artists.py are:

```bash
--training-path #specify path to training data
--validation-path #specify path to validation data
--epochs #specify amount of epochs, default = 10
--image-size #specify image size, default H200W200. Larger sizes will slow down the computation. 
--learning-rate #Specify learning rate, default = 0.1
--hidden-layers #Specify neurons in hidden layer, default = 32.
--output # specify output path
```

Using all the parameters:
```bash
python cnn-artists.py --training-path impressionist/training --validation-path impressionist/validation --output output --epochs 5 --image-size H300W300 --learning-rate 0.001 --hidden-layers 16
```


