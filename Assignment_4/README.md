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
    <img src="/Assignment_4/outputs/heatmap.png" alt="Logo" width="500" height="350">
</a>


# User Defined Parameters
The user defined arguments for lr-mnist.py are:

```bash
--mnist #specify either path to mnist dataset or download to use fetch_openml to download the whole dataset.
--output #Path where you want the output files.
--solver # specify solver algorithm - can be 'newton-cg', ‘sag’, ‘saga’ and ‘lbfgs’. default='saga'.
--penalty # Specify norm used in penalization, can be 'l2' or None. Default=None.
--test_image # Path to an image on which you wish to test the model.
 
```


The user defined arguments for nn-mnist.py are:
```bash
--mnist #specify either path to mnist dataset or download to use fetch_openml to download the whole dataset.
--output #Path where you want the output files.
--layers # specify hidden layers, default = 32 16.
--test_split #Spcifies train/test split, default = 0.2.
--epochs # Specify amount of epochs, default = 100.
--save_model_path # Path to which you want to save the trained model.
--test_image # Path to an image on which you wish to test the model.
```


Using all the parameters:
```bash
python src/nn-mnist.py --mnist download --test-split 0.1 -- layers 16 8 --test_image test.png --output outputs --save_model_path outputs
```

# Loading saved model

When using save_model_path, the script will output a trained model file called nn_model.pkl. <b\>
To reuse the model, use joblib.load to load the model from the filename.
```bash
import os
import joblib

loaded_model = joblib.load(os.path.join("path_to_dir", "nn_model.pkl"))
result = loaded_model.score(X_test, Y_test)
```

