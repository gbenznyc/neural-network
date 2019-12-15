import numpy as np

class Perceptron:
	def __init__(self, input_nodes, learning_rate, epochs, one_output):
		self.input_nodes = input_nodes
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.one_output = one_output

		#Initialize weights, we add one to the input_nodes
		#to account for the extra bias node we will be including
		self.weights = np.random.uniform(low=-1.0, high=1.0, size=(input_nodes + 1,))
		print(self.weights)

	def activation_function(self, x):
		#We will be using a sigmoid function as our activation_function
		return 1 / (1 + np.exp(-x))

	def predict(self, inputs):
		numpy_inputs = np.asarray(inputs)

		#Add the bias node to the input layer
		numpy_inputs = np.insert(numpy_inputs, 0, 1)

		#Calculate the summation of all of the inputs with the weights
		weighted_sum = np.dot(numpy_inputs, self.weights)

		#Use return the weighted_sum passed through the activation function
		if one_output:
			#If we only have one output node multiply value by 10
			return self.activation_function(weighted_sum) * 10
		else:
			#If we have 10 output nodes, keep the value between 0 and 1
			return self.activation_function(weighted_sum)


	def learn(self, training_data):
		#Iterate through amount of epochs
		for i in range(0, epochs):
			for data in training_data:
				#Predict a result based on some data
				prediction = self.predict(data["data"])

				#Find the difference between the prediction and the actual label
				error = data["answer"] - prediction

				self.weights = 



def main():
	test = Perceptron(10, 0, 0, one_output=True)
	print(test.predict([1,2,3,4,5,6,7,8,9,10]))

main()