# Neural Net

The Python 3.6 code in this project is a neural net that is used for digit identification. Digits are provided to us in two formats. Info can be found on them in the digit-recognition-examples folder in the 32x32-bitmaps folder and the 8x8-integer-inputs folder. Our neural network uses perceptrons to predict what these digits actually are. The specifics of our implementation and the mechanics behind our neural network can be read more about in our lab report.

To run our code one must first ensure they have Python 3.6 installed on there computer. Next the neccesary libraries must be installed in the requirements.txt file. 

Our neural net can be run be typing '''python3 main.py input_rep output_rep learn_rate'''
input_rep = size of input representation, must be 8 or 32 (int)
num_output_nodes = number of output nodes, must be 1 or 10 (int)
learn_rate = learning rate, recommended 0 < learn_rate <= 1 (double)

As the algorithm runs, the program outputs the proportion of correct digits classified after the end of each epoch.
