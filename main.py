from perceptron import Perceptron
from multiple_perceptrons import MultiplePerceptrons
from reader import Reader

def main():
	train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
	train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()

	test_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes").return_data()
	test_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes").return_data()

	print(train_8[0])

	gn = MultiplePerceptrons(1024, 0.1, 25)
	gn.learn(train_32, test_32)
	

main()