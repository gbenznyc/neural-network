from perceptron import Perceptron
import numpy as np

class MultiplePerceptrons:
	def __init__(self, input_nodes, learning_rate, epochs, output_nodes=10):
		self.input_nodes = input_nodes
		self.learning_rate = learning_rate
		self.epochs = epochs

		#Initialize group of perceptrons
		self.perceptrons = []
		for i in range(output_nodes):
			self.perceptrons.append(Perceptron(self.input_nodes, self.learning_rate))

	def predict(self, inputs):
		#Run inputs through all perceptrons
		results = []
		for perceptron in self.perceptrons:
			results.append(perceptron.predict(inputs))

		numpy_results = np.asarray(results)

		#Find the node with the largest prediction value
		max_index = np.where(numpy_results == np.amax(numpy_results))[0][0]
		
		return max_index

	def learn(self, training_data):
		#Train the NN based on the amount of epochs
		for i in range(self.epochs):
			for data in training_data:
				#Find the max value node 
				prediction = self.predict(data["data"])
				if prediction == data["answer"]:
					error = 1
				else:
					error = 0
		