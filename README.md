<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Guscode/cds-visual-exam">
    <img src="README_images/styletransfer.jpeg" alt="Logo" width="343" height="170">
  </a>
  
  <h1 align="center">Cultural Data Science 2021</h1> 
  <h3 align="center">Visual Analytics Exam</h3> 


  <p align="center">
    Gustav Aarup Lauridsen 
    <br />
  <p align="center">
    ID: au593405 
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Exam-information">Exam Info</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li>
      <a href="#portfolio-assignments">Portfolio assignments</a>
      <ul>
        <li><a href="#assignment-3---edge-detection">Assignment 3 - Edge detection</a></li>
        <li><a href="#assignment-4---logistic-regression-and-neural-network-benchmark-mnist-classification">Assignment 4 - Logistic Regression and Neural Network benchmark mnist classification</a></li>
        <li><a href="#assignment-5---cnn-classification-of-impressionist-paintings">Assignment 5 - CNN classification of impressionist paintings</a></li>
        <li><a href="#self-assigned-project">self-assigned project</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- PROJECT INFO -->
## Project info

This repository contains assignments and descriptions regarding an exam in cultural data science at Aarhus Univerisy [_Visual Analytics_](https://kursuskatalog.au.dk/en/course/101992/Visual-Analytics). The three class assignments are included in this repository, and the self-assigned project is found at LINK. 

The class assignments included in this portfolio are:
* Assignment 3 - _Edge detection_
* Assignment 4 - _Logistic Regression and Neural Network benchmark mnist classification_
* Assignment 5 - _CNN classification of impressionist paintings_

<!-- HOW TO RUN -->
## How to run

To run the assignments, you need to go through the following steps in your bash-terminal to configure a virtual environment on Worker02 (or your local machine) with the needed prerequisites for the class assignments:

__Setting up virtual environment and downloading data__
```bash
cd {directory where you want the assignment saved)
git clone https://github.com/Guscode/cds-visual-exam.git
cd cds-visual-exam
bash create_vis_venv.sh
source visvenv/bin/activate
```

### Assignment 3 - Edge detection

Go through the following steps to run assignment 3:

This code will perform edge detection on a cropped version of jefferson.jpg and save the output image and text in /output.

```bash
cd assignment_3
python3 detect_edges.py --image jefferson.jpg --crop-coordinates X750X700Y750Y1150 --psm 5 --output output
```

This code will perform edge detection on the images in the folder signs and save the output in /output.
```bash
cd assignment_3
python3 detect_edges.py --image-files signs --output output
```

Type: ```python3 edge_detection.py -h``` for a detailed guide on how to specify script-parameters. 


For details and results see [```assignment_3/README.md```](https://github.com/Guscode/cds-visual-exam/tree/main/assignment_3)

