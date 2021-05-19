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


