from perceptron import Perceptron
from multiple_perceptrons import MultiplePerceptrons
from reader import Reader

def main():
	train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
	train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()


	gn = MultiplePerceptrons(64, 0.1, 5)
	gn.predict(train_8[1]["data"])

	# nn = Perceptron(64, 0.1, 5, True)
	# nn.learn(train_8)
	# print("TEST")
	# print(train_8[3]["answer"])
	# print(nn.predict(train_8[1]["data"]))
	# print(train_8[4]["answer"])
	# print(nn.predict(train_8[4]["data"]))
	# print(train_8[5]["answer"])
	# print(nn.predict(train_8[5]["data"]))

main()