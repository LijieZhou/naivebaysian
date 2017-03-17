import csv
import math

def loadCsv(filename):
	lines = csv.reader(open(filename,"rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [int(x) for x in dataset[i]]
	return dataset

# filename = 'PilotTest_numerical.csv'
# dataset = loadCsv(filename)
# print('Loaded data file {0} with {1} rows').format(filename,len(dataset))


# The seperateByClass funciton will break the whole dataset based on their class value
# The outcome is N groups with N different class values. 

def seperateByClass(dataset):
	seperated = {} #dict
	for i in range(len(dataset)): # iterate each row in the dataset
		vector = dataset[i]	      
		if(vector[-1] not in seperated): # if the row is not in the seperated dictionary
			seperated[vector[-1]] = []   # Create a new entry in the corresponding seperated class 
		seperated[vector[-1]].append(vector)  # append the row to its corresponding seperated class
	return seperated   

# filename = 'PilotTest_numerical.csv'
# dataset = loadCsv(filename)
# seperated = seperateByClass(dataset)
# print('Seperated instances: {0}').format(seperated)

def mean(numbers):
	return sum(numbers)/len(numbers)

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float((len(numbers)-1))
	return math.sqrt(variance)

# numbers = [1,2,3,4,5]
# print('Summary of {0}: mean={1},stdev={2}').format(numbers,mean(numbers),stdev(numbers))


def summarize(dataset):
	summaries = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)] 
	del summaries[-1] # remove the last column since we do not need the class itself
	return summaries 

# filename = 'PilotTest_numerical.csv'
# dataset = loadCsv(filename)
# summary = summarize(dataset)
# print('Attribute summaries: {0}').format(summary)


def summarizeByClass(dataset):
	seperated = seperateByClass(dataset)
	summaries = {}

	for classValue, instances in seperated.items():
		summaries[classValue] = summarize(instances)
	return summaries

# filename = 'PilotTest_numerical.csv'
# dataset = loadCsv(filename)
# summaries = summarizeByClass(dataset)
# print('Summary by class value: {0}').format(summary)


# Next, we use the Gaussian Probability Density Function (PDF) to estinate the probability of a given attribute value,
# given the known mean and standard deviation for the attribute

def calculatePDF(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# Simple test with a given attribute value x
# x = 71.5
# mean = 73
# stdev = 6.2
# probability = calculatePDF(x,mean,stdev)
# print('Probability of belonging to this class: {0}').format(probability)

# Calculate Class Probablities
# The goal is to combine the probabilities of all the attribute values for a data instance
# and come up with a probability of the entire data instance belonging to the class
def calculateClassProbabilities(summaries,inputVector):
	probabilities = {}  # map 
	for classValue, classSummaries in summaries.items(): # Iterature the summaries map
		probabilities[classValue] = 1				     # Set the default probability to 1 
		for i in range(len(classSummaries)): 
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculatePDF(x,mean,stdev)
	return probabilities

#simple test for calculateClassProbabilities
# summaries = {0: [(39, 10.535653752852738), (174391, 88638.48194174751), (9, 3.2307119958300214), (127, 527.2732332481899), (120, 495.25776117088765), (36, 10.720890821195784)],1:[(41, 9.172941575244941), (192397, 79103.12997509734), (12, 2.390457218668787), (2407, 5053.890495732683), (1, 0.1), (50, 13.887301496588272)]}
# inputVector = [31,45781,14,14084,1,50]
# probabilities = calculateClassProbabilities(summaries,inputVector)
# print('Probabilities for each class: {0}').format(probabilities)

def predict(summaries,inputVector):
	probabilities = calculateClassProbabilities(summaries,inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries,testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries,testSet[i])
		predictions.append(result)
	return predictions
# Quick test for getPredictions
# summaries = {0: [(36, 13.479209907676942), (0, 0.14463027703223794), (9, 2.494291806341218), (0, 0.43470374203159945), (0, 0.9180690815412021), (0, 0.7857119932108553), (148, 936.2516760137105), (53, 310.22405777021464), (39, 11.95800794882443)], 1: [(43, 10.314328560369736), (0, 0.2827108115322792), (11, 2.4448356304989476), (0, 0.7065885298197898), (0, 0.9544717830861247), (0, 0.923041058417326), (3937, 14386.060035041686), (193, 592.8260649686782), (45, 10.7602139047949)]}
# testset = [[54,0,9,0,0,0,0,0,20],[25,0,7,0,0,1,0,0,40]]
# predictions = getPredictions(summaries,testset)
# print('Predictions:{0}').format(predictions)


def main():
	trainingFilename = 'PilotTest_numerical.csv'
	trainingDataset = loadCsv(trainingFilename)
	testFilename = 'test_analytical.csv'
	testDataset = loadCsv(testFilename)
	# print('Loaded data file {0} with {1} rows').format(testFilename,len(testDataset))


	#prepare model
	summaries = summarizeByClass(trainingDataset)
	# inputVector = [54,0,9,0,0,0,0,0,20]
	# result = predict(summaries,inputVector)
	predictions = getPredictions(summaries,testDataset)
	print('Prediction: {0}').format(predictions)

main()


# summaries = {0: [(36, 13.479209907676942), (0, 0.14463027703223794), (9, 2.494291806341218), (0, 0.43470374203159945), (0, 0.9180690815412021), (0, 0.7857119932108553), (148, 936.2516760137105), (53, 310.22405777021464), (39, 11.95800794882443)], 1: [(43, 10.314328560369736), (0, 0.2827108115322792), (11, 2.4448356304989476), (0, 0.7065885298197898), (0, 0.9544717830861247), (0, 0.923041058417326), (3937, 14386.060035041686), (193, 592.8260649686782), (45, 10.7602139047949)]}
# inputVector = [54,0,9,0,0,0,0,0,20]
# result = predict(summaries,inputVector)
# print('Prediction: {0}').format(result)




