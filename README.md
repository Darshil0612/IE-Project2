Word Recognizer Project

Description

This Python project utilizes Hidden Markov Models (HMM) for recognizing words from acoustic signals. It is designed to process labeled speech data, learning to recognize and predict words based on training data.


Required Python libraries: numpy, scipy, matplotlib, tqdm, pickle, collections, itertools, string, os.

Data Files:

The project requires the following data files, which should be placed in the current working directory:

clsp.lblnames - Contains label names.
clsp.trnlbls - Training labels.
clsp.endpts - Endpoints for training segments.
clsp.trnscr - Transcriptions of training words.
clsp.devlbls - Development set labels.

Structure and Modules:

The script contains several classes and functions essential for the training and operation of the HMM:

HMM class: Defines the structure and operations of a Hidden Markov Model.

Word_Recognizer class: Manages the entire process of word recognition including data preparation, training, and testing of the model.
Main function to drive the training process.

Usage
To train and test the model, navigate to the local directory containing the script and data files after changing the directory in the code to run it locally, then run:

python project2.py
contrastive.py


NOTES:
I have taken help from the existing HMM on piazza but made significant changes and most of the code is my own.
I have implemented the primary system but cannot successfully debug it and have implmented every part of the assignmnet.
I have built a contastive system outline but cannot completely execute it.
Discussed the project with Janvi Prasad and Hirtika Mirghani.
