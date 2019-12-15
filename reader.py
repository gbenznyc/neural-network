class Reader:
	def __init__(self, filename):
		self.filename = filename
		self.data = []

		#Case where we are using 8x8
		if "8" in self.filename:
			with open(self.filename) as f: 
				for line in f:
					digits = line.split(",")
					temp_dict = {"data": [int(i) for i in digits[:-1]], "answer": int(digits[-1:][0].strip())}
					self.data.append(temp_dict)

		#Case where we are using 32x32
		if "32" in self.filename:
			with open(self.filename) as f: 
				temp_list = []
				for line in f:
					if line[0] == ' ':

						temp_dict = {"data": temp_list, "answer": int(line[1])}
						self.data.append(temp_dict)
						temp_list = []
					else:
						temp_list += [int(i) for i in line.strip()]


def main():
	test = Reader("digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes")
	test2 = Reader("digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes")
	#print(test.data)

#main()