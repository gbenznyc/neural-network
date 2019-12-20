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
		#Run inputs through all perceptrons and append to an array
		results = []
		for perceptron in self.perceptrons:
			results.append(perceptron.predict(inputs))

		numpy_results = np.asarray(results)

		#Find the node with the largest prediction value from the array
		max_index = np.where(numpy_results == np.amax(numpy_results))[0][0]

		#Return the results array and the index of the greatest value perceptron
		return max_index, numpy_results

	def learn(self, training_data, testing_data):
		#Train the NN based on the amount of epochs
		for i in range(self.epochs):
			print("Current epoch: " + str(i))
			for data in training_data:
				#Find the max value node as the prediction
				prediction, numpy_results = self.predict(data["data"])

				#For every perceptron, check to see if it is the correct value
				for index, perceptron in enumerate(self.perceptrons):
					output = numpy_results[index]
					
					#If the prediction is the same as the answer AND
					#the specific nodes index matches the prediction
					if prediction == index and prediction == data["answer"]:
						error = 1 - output
					else:
						error = 0 - output

					#Update that specfic perceptron's weights
					perceptron.learn(data, output, error=error)
			
			#Testing each epoch
			correct = 0
			for data in testing_data:
				if self.predict(data["data"])[0] == data["answer"]:
					correct += 1

			print("Accuracy: " + str(correct/len(testing_data)))

		