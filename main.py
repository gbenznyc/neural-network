from perceptron import Perceptron
from multiple_perceptrons import MultiplePerceptrons
from reader import Reader

def main():
	train_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra").return_data()
	train_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra").return_data()

	test_32 = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes").return_data()
	test_8 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes").return_data()

	gn = MultiplePerceptrons(64, 0.1, 5)
	gn.learn(train_8)
	correct = 0
	for data in test_8:

		if gn.predict(data["data"])[0] == data["answer"]:
			correct += 1

	print(correct/len(test_8))

	
	# nn = Perceptron(1024, 0.1, 10, one_output=True)
	# nn.learn(train_32)
	# correct = 0
	# for data in test_32:
	# 	#print("data[answer]: ") #debug
	# 	#print(data["answer"]) #debug
	# 	print("nn.predict(data[data]): ") #debug
	# 	print(nn.predict(data["data"])) #debug
	# 	if nn.predict(data["data"]) == data["answer"]:
	# 		correct += 1


main()