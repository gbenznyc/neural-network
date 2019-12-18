from perceptron import Perceptron
from multiple_perceptrons import MultiplePerceptrons
from reader import Reader

def main():
	train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
	train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()

	test_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes").return_data()
	test_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes").return_data()

	gn = MultiplePerceptrons(64, 0.1, 1)
	#gn.learn(train_8)
	correct = 0
	for data in test_8:
		print(gn.predict(data["data"]))
		if gn.predict(data["data"]) == data["answer"]:
			correct += 1

	print(correct/len(test_8))


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