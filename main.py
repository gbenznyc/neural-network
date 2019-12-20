import sys
from perceptron import Perceptron
from multiple_perceptrons import MultiplePerceptrons
from reader import Reader
	
train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()

test_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes").return_data()
test_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes").return_data()

def main():
	#read in parameters
	input_rep = int(sys.argv[1])
	num_output_nodes = int(sys.argv[2])
	learn_rate = float(sys.argv[3])

	def print_help():
		print("Run the neural net as follows:")
		print("python3 main.py input_rep output_rep learn_rate")
		print("input_rep         = size of input representation, must be 8 or 32 (int)")
		print("num_output_nodes  = number of output nodes, must be 1 or 10 (int)")
		print("learn_rate        = learning rate, recommended 0 < learn_rate <= 1 (double)")

	#quit if input or output representations have illegal values
	if input_rep != 8 and input_rep != 32:
		print_help()
		quit()

	if num_output_nodes != 1 and num_output_nodes != 10:
		print_help()
		quit()

	num_input_nodes = input_rep ** 2

	if num_output_nodes == 1:
		nn = Perceptron(num_input_nodes, learn_rate, 50, True)
	else:
		nn = MultiplePerceptrons(num_input_nodes, learn_rate, 50)

	if input_rep == 8:
		nn.learn(train_8, test_8)
	else:
		nn.learn(train_32, test_32)

	print("Training finished. 50 epochs completed.")

main()