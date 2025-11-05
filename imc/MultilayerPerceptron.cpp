/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include <cstdlib>
#include <ctime>

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{

}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	srand(time(nullptr));

	//  store number of layers
	nOfLayers = nl;

	// allocate memory
	layers = new Layer[nOfLayers];
	for (int i = 0; i < nOfLayers; i++)
	{
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[npl[i]];

		for (int j = 0; j < npl[i]; j++)
		{
			layers[i].neurons[j].out = 0.0;
			layers[i].neurons[j].delta = 0.0;
			//  input layer will be at index 0 
			if (i==0)
			{
				layers[i].neurons[j].w = nullptr;
				layers[i].neurons[j].deltaW = nullptr;
				layers[i].neurons[j].lastDeltaW = nullptr;
				layers[i].neurons[j].wCopy = nullptr;
			}else{
				// for each layer number of inputs will be equalto neurons in previous layer + 1 for bias
				int numInputs = npl[i-1] + 1;
				
				// create empty vectors for storing values for each field of Neuron
				layers[i].neurons[j].w = new double[numInputs];
				layers[i].neurons[j].deltaW = new double[numInputs];
				layers[i].neurons[j].lastDeltaW = new double[numInputs];
				layers[i].neurons[j].wCopy = new double[numInputs];
				// then for each of them set initial value of 0.0
				for (int k = 0; k < numInputs; k++)
				{	
					layers[i].neurons[j].w[k] = 0.0;
					layers[i].neurons[j].deltaW[k] = 0.0;
					layers[i].neurons[j].lastDeltaW[k] = 0.0;
					layers[i].neurons[j].wCopy[k] = 0.0;
				}
				

			}

			
			
		}
		
	}
	randomWeights();
	

	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}



// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
    if (layers == nullptr) return;
    for (int i = 0; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            delete[] layers[i].neurons[j].w;
            delete[] layers[i].neurons[j].deltaW;
            delete[] layers[i].neurons[j].lastDeltaW;
            delete[] layers[i].neurons[j].wCopy;
        }
        delete[] layers[i].neurons;
    }
    delete[] layers;
    layers = nullptr;
}


// ------------------------------
// fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
for (int i = 1; i < nOfLayers; i++) // start from 1 because we dont initialize for input layer
{
	for (int j = 0; j < layers[i].nOfNeurons; j++)
	{
		int nInputs = layers[i-1].nOfNeurons +1;
		for (int k = 0; k < nInputs; k++)
		{
			layers[i].neurons[j].w[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
		}
		
	}
	
}

}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for (int i = 0; i < layers[0].nOfNeurons; i++)
	{
		layers[0].neurons[i].out = input[i]; // set inputs of model as output for first layer for future feedforward
	}
	
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	Layer &outputLayer = layers[nOfLayers - 1];
	for (int i = 0; i < outputLayer.nOfNeurons; i++)
	{
		output[i] = outputLayer.neurons[i].out;
	}
	
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            int nInputs = layers[i-1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++)
                layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
        }
    }
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            int nInputs = layers[i-1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++)
                layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
        }
    }
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double net = 0.0;

			int nInputs = layers[i-1].nOfNeurons;
			for (int k = 0; k < nInputs; k++)
			{
				net += layers[i].neurons[j].w[k] * layers[i-1].neurons[k].out; //perform w * x operation for every weight in every layers every neuron and then save summary of them
			}
			
			// add bias(last in the weights)
			net += layers[i].neurons[j].w[nInputs] * 1.0;

			// apply activation function (sigmoid)
			layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net));

		}
		
	}
	
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
	double meanSquareError = 0.0;

	Layer &outputLayer = layers[nOfLayers - 1];
	for (int i = 0; i < outputLayer.nOfNeurons; i++)
	{
		double diff = target[i] - outputLayer.neurons[i].out; // we calculate difference between targets and last layer's predictions
		meanSquareError += diff * diff; // power
	}
	return meanSquareError / outputLayer.nOfNeurons; 
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	Layer &outputLayer = layers[nOfLayers -1];
	// these are output layers deltas
	for (int i = 0; i < outputLayer.nOfNeurons; i++)
	{
		double out = outputLayer.neurons[i].out;
		double error = target[i] - out;
		outputLayer.neurons[i].delta = error * out * (1.0 - out);
	}
	// these hidden layer's
	// we should go into opposite direction of network to calculate from the end(excluding output layer)
	for (int i = nOfLayers - 2; i > 0; i--)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double out = layers[i].neurons[j].out;
			double sum = 0.0;

			for (int k = 0; k < layers[i+1].nOfNeurons; k++)
			{
				sum += layers[i+1].neurons[k].delta * layers[i+1].neurons[k].w[j];
			}
			layers[i].neurons[j].delta = out * (1.0 - out) * sum;
			
		}
		
	}
	
	
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {

}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
    for (int i = 1; i < nOfLayers; i++) { // skip input layer
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            Neuron& neuron = layers[i].neurons[j];
            int nInputs = layers[i - 1].nOfNeurons;

            for (int k = 0; k < nInputs; k++) {
                double outPrev = layers[i - 1].neurons[k].out;
                double newDeltaW = eta * neuron.delta * outPrev + mu * neuron.lastDeltaW[k];
                neuron.w[k] += newDeltaW;

                neuron.lastDeltaW[k] = newDeltaW;
            }

            // bias weight
            double newDeltaWbias = eta * neuron.delta * 1.0 + mu * neuron.lastDeltaW[nInputs];
            neuron.w[nInputs] += newDeltaWbias;
            neuron.lastDeltaW[nInputs] = newDeltaWbias;
        }
    }
}


// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
    for (int i = 1; i < nOfLayers; i++) {
        cout << "Layer " << i << " weights:" << endl;
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            cout << " Neuron " << j << ": ";
            int nInputs = layers[i - 1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++) {
                cout << layers[i].neurons[j].w[k] << " ";
            }
            cout << endl;
        }
    }
}

// old functions
// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// void MultilayerPerceptron::performEpochOnline(double* input, double* target) {
//     feedInputs(input);
//     forwardPropagate();
//     backpropagateError(target);
//     weightAdjustment();
// }


// // ------------------------------
// // Perform an online training for a specific trainDataset
// void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
// 	int i;
// 	for(i=0; i<trainDataset->nOfPatterns; i++){
// 		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
// 	}
// }

// // ------------------------------
// // Test the network with a dataset and return the MSE
// double MultilayerPerceptron::test(Dataset* testDataset) {
//     double totalError = 0.0;
//     double* prediction = new double[testDataset->nOfOutputs];

//     for (int i = 0; i < testDataset->nOfPatterns; i++) {
//         feedInputs(testDataset->inputs[i]);
//         forwardPropagate();
//         getOutputs(prediction);
//         totalError += obtainError(testDataset->outputs[i]);
//     }

//     delete[] prediction;
//     return totalError / testDataset->nOfPatterns;
// }
// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
// void MultilayerPerceptron::predict(Dataset* pDatosTest)
// {
// 	int i;
// 	int j;
// 	int numSalidas = layers[nOfLayers-1].nOfNeurons;
// 	double * obtained = new double[numSalidas];
	
// 	cout << "Id,Predicted" << endl;
	
// 	for (i=0; i<pDatosTest->nOfPatterns; i++){

// 		feedInputs(pDatosTest->inputs[i]);
// 		forwardPropagate();
// 		getOutputs(obtained);
		
// 		cout << i;

// 		for (j = 0; j < numSalidas; j++)
// 			cout << "," << obtained[j];
// 		cout << endl;

// 	}
// }
// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
// {
// 	int countTrain = 0;

// 	// Random assignment of weights (starting point)
// 	randomWeights();

// 	double minTrainError = 0;
// 	int iterWithoutImproving;
// 	double testError = 0;

// 	// Learning
// 	do {

// 		trainOnline(trainDataset);
// 		double trainError = test(trainDataset);
// 		if(countTrain==0 || trainError < minTrainError){
// 			minTrainError = trainError;
// 			copyWeights();
// 			iterWithoutImproving = 0;
// 		}
// 		else if( (trainError-minTrainError) < 0.00001)
// 			iterWithoutImproving = 0;
// 		else
// 			iterWithoutImproving++;

// 		if(iterWithoutImproving==50){
// 			cout << "We exit because the training is not improving!!"<< endl;
// 			restoreWeights();
// 			countTrain = maxiter;
// 		}


// 		countTrain++;

// 		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

// 	} while ( countTrain<maxiter );

// 	cout << "NETWORK WEIGHTS" << endl;
// 	cout << "===============" << endl;
// 	printNetwork();

// 	cout << "Desired output Vs Obtained output (test)" << endl;
// 	cout << "=========================================" << endl;
// 	for(int i=0; i<pDatosTest->nOfPatterns; i++){
// 		double* prediction = new double[pDatosTest->nOfOutputs];

// 		// Feed the inputs and propagate the values
// 		feedInputs(pDatosTest->inputs[i]);
// 		forwardPropagate();
// 		getOutputs(prediction);
// 		for(int j=0; j<pDatosTest->nOfOutputs; j++)
// 			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
// 		cout << endl;
// 		delete[] prediction;

// 	}

// 	testError = test(pDatosTest);
// 	*errorTest=testError;
// 	*errorTrain=minTrainError;

// }
// old functions

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;


	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		double trainError = test(trainDataset);
		if(countTrain==0 || trainError < minTrainError){
			if( (minTrainError-trainError) > 0.00001)
			    iterWithoutImproving = 0;
			else
			    iterWithoutImproving++;
			minTrainError = trainError;
			copyWeights();
		}
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (j==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
