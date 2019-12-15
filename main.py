from perceptron import Perceptron
from reader import Reader

def main():
	train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
	train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()

	nn = Perceptron(1024, 0.01, 5, True)
	nn.learn(train_32)
	print(train_32[3]["answer"])
	print(nn.predict(train_32[3]["data"]))
	print(train_32[4]["answer"])
	print(nn.predict(train_32[4]["data"]))
	print(train_32[5]["answer"])
	print(nn.predict(train_32[5]["data"]))

main()