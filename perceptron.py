import numpy as np

class Perceptron:
	def __init__(self, input_nodes, learning_rate, epochs=50, one_output=False):
		self.input_nodes = input_nodes
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.one_output = one_output

		#Initialize weights, we add one to the input_nodes
		#to account for the extra bias node we will be including
		self.weights = np.random.uniform(low=-0.15, high=0.15, size=(input_nodes + 1,))

	def activation_function(self, x):
		#We will be using a sigmoid function as our activation_function
		return 1 / (1 + np.exp(-x + 0.5))

	def predict(self, inputs):
		numpy_inputs = np.asarray(inputs)

		#Add the bias node to the input layer
		numpy_inputs = np.insert(numpy_inputs, 0, 1)

		#Calculate the summation of all of the inputs with the weights
		weighted_sum = np.dot(numpy_inputs, self.weights)

		#Return the weighted_sum passed through the activation function
		if self.one_output:
			#If we only have one output node multiply value by 10
			sigmoid_val = self.activation_function(weighted_sum)
			if sigmoid_val <= 0.1:
				return 0, sigmoid_val
			elif sigmoid_val <= 0.2:
				return 1, sigmoid_val
			elif sigmoid_val <= 0.3:
				return 2, sigmoid_val
			elif sigmoid_val <= 0.4:
				return 3, sigmoid_val
			elif sigmoid_val <= 0.5:
				return 4, sigmoid_val
			elif sigmoid_val <= 0.6:
				return 5, sigmoid_val
			elif sigmoid_val <= 0.7:
				return 6, sigmoid_val
			elif sigmoid_val <= 0.8:
				return 7, sigmoid_val
			elif sigmoid_val <= 0.9:
				return 8, sigmoid_val
			else:
				return 9, sigmoid_val
		else:
			#If we have 10 output nodes, keep the value between 0 and 1
			return self.activation_function(weighted_sum)


	def learn(self, training_data, testing_data=None, output=0, error=0):
		#Iterate through amount of epochs individually
		if self.one_output:
			for i in range(0, self.epochs):
				print("Current epoch: " + str(i))
				for data in training_data:
					#Predict a result based on some data
					prediction, output = self.predict(data["data"])

					#Find the difference between the prediction and the actual label
					error = (data["answer"] - (output * 10.0))/10

					#Update the weight of the bias node
					self.weights[0] += error * self.learning_rate

					#Update the weight of all other inputs
					self.weights[1:] += error * self.learning_rate * np.asarray(data["data"])
					
				#Testing each epoch
				correct = 0
				for data in testing_data:
					if self.predict(data["data"])[0] == data["answer"]:
						correct += 1

				print("Accuracy: " + str(correct/len(testing_data)))
		
		#If the perceptron is part of a group of others
		else:
			#Update the weight of the bias node
			self.weights[0] += error * self.learning_rate 

			#Update the weight of all other inputs
			self.weights[1:] += error * self.learning_rate * np.asarray(training_data["data"]) 
