This repository implements a word-level lip reading approach using a CNN-LSTM architecture. The model and preprocessing pipeline are designed for the German Lip Reading Dataset (GLips). The model aims to classify short video sequences of spoken german words.

 ## Usage
 
1. Install dependencies
2. Change the paths in `settings.py` matching your file structure
3. Run `python preprocess_images.py` to preprocess the images, convert them to `.npy` and compute mean and standard deviation
4. Run `python main.py` to train the model. If you want to change model parameters, do so in `settings.py`
5. While training the loss and accuracy will be printed and the model state is saved after each epoch

 ## Dataset

This project uses the German Lip Reading Dataset (GLips) dataset, available [[here](https://www.fdr.uni-hamburg.de/record/10048)].

**Citation:**

> Gerald Schwiebert, Cornelius Weber, Leyuan Qu, Henrique
Siqueira, and Stefan Wermter. 2022. GLips - German Lipreading Dataset. University of Hamburg. [![DOI](https://www.fdr.uni-hamburg.de/badge/DOI/10.25592/uhhfdm.10048.svg)](https://doi.org/10.25592/uhhfdm.10048)
