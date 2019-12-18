import numpy as np

class Perceptron:
	def __init__(self, input_nodes, learning_rate, epochs=50, one_output=False):
		self.input_nodes = input_nodes
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.one_output = one_output

		#Initialize weights, we add one to the input_nodes
		#to account for the extra bias node we will be including
		self.weights = np.random.uniform(low=-1.0, high=1.0, size=(input_nodes + 1,))

	def activation_function(self, x):
		#We will be using a sigmoid function as our activation_function
		return 1 / (1 + np.exp(-x + 0.5))

	def sigmoid_deriviate(self, x):
		#The deriviate of the sigmoid used to minimze our error when learning
		return self.activation_function(x)*(1- self.activation_function(x))

	def predict(self, inputs):
		numpy_inputs = np.asarray(inputs)

		#Add the bias node to the input layer
		numpy_inputs = np.insert(numpy_inputs, 0, 1)

		#Calculate the summation of all of the inputs with the weights
		weighted_sum = np.dot(numpy_inputs, self.weights)

		#Return the weighted_sum passed through the activation function
		if self.one_output:
			#If we only have one output node multiply value by 10
			#print(self.activation_function(weighted_sum))
			sigmoid_val = self.activation_function(weighted_sum)
			if sigmoid_val <= 0.05:
				return 0
			elif sigmoid_val <= 0.15:
				return 1
			elif sigmoid_val <= 0.25:
				return 2
			elif sigmoid_val <= 0.35:
				return 3
			elif sigmoid_val <= 0.45:
				return 4
			elif sigmoid_val <= 0.55:
				return 5
			elif sigmoid_val <= 0.65:
				return 6
			elif sigmoid_val <= 0.75:
				return 7
			elif sigmoid_val <= 0.85:
				return 8
			else:
				return 9
		else:
			#If we have 10 output nodes, keep the value between 0 and 1
			return self.activation_function(weighted_sum)


	def learn(self, training_data, output=0, error = 0):
		#Iterate through amount of epochs individually
		if self.one_output:
			for i in range(0, self.epochs):
				for data in training_data:
					#Predict a result based on some data
					prediction = self.predict(data["data"])

					#Find the difference between the prediction and the actual label
					#Need to review if this error actually does anything for one output node
					error = data["answer"] - prediction

					#Update the weight of the bias node
					self.weights[0] += error * self.learning_rate

					#Update the weight of all other inputs
					self.weights[1:] += error * self.learning_rate * np.asarray(data["data"])
		
		#If the perceptron is part of a group of others
		else:
			#Update the weight of the bias node
			self.weights[0] += error * self.learning_rate * output*(1-output)

			#Update the weight of all other inputs
			self.weights[1:] += error * self.learning_rate * np.asarray(training_data["data"]) * output*(1-output)


def main():
	test = Perceptron(10, 0, 0, one_output=True)
	print(test.predict([1,2,3,4,5,6,7,8,9,10]))

#main()