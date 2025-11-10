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
	eta = 0.7;          // assignment-recommended default
	mu = 1.0;           // assignment-recommended default
	online = false;     // default to offline (batch)
	outputFunction = 0; // 0=sigmoid, 1=softmax (default is sigmoid)
}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
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
				layers[i].neurons[j].deltaW[k] = 0.0;       // clear accumulators (batch)
				layers[i].neurons[j].lastDeltaW[k] = 0.0;   // clear momentum
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
				net += layers[i].neurons[j].w[k] * layers[i-1].neurons[k].out;
			}
			
			// add bias
			net += layers[i].neurons[j].w[nInputs] * 1.0;

			// Hidden layers: sigmoid
			// Output layer: sigmoid if outputFunction==0; otherwise store net to apply softmax after loop
			if (i < nOfLayers - 1 || outputFunction == 0) {
				layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net));
			} else {
				layers[i].neurons[j].out = net; // temporary store pre-activation for softmax
			}
		}

		// Output layer softmax with numerical stability
		if (i == nOfLayers - 1 && outputFunction == 1) {
			int m = layers[i].nOfNeurons;
			double maxNet = layers[i].neurons[0].out;
			for (int j = 1; j < m; j++)
				if (layers[i].neurons[j].out > maxNet) maxNet = layers[i].neurons[j].out;

			double denom = 0.0;
			for (int j = 0; j < m; j++)
				denom += exp(layers[i].neurons[j].out - maxNet);

			for (int j = 0; j < m; j++)
				layers[i].neurons[j].out = exp(layers[i].neurons[j].out - maxNet) / denom;
		}
	}
}

// ------------------------------
// Obtain the output error (MSE or Cross-Entropy) averaged over outputs
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {
	Layer &outputLayer = layers[nOfLayers - 1];
	int k = outputLayer.nOfNeurons;

	if (errorFunction == 0) {
		double mse = 0.0;
		for (int i = 0; i < k; i++) {
			double diff = target[i] - outputLayer.neurons[i].out;
			mse += diff * diff;
		}
		return mse / k;
	} else {
		const double eps = 1e-12;
		double ce = 0.0;
		for (int i = 0; i < k; i++) {
			double o = outputLayer.neurons[i].out;
			if (o < eps) o = eps;
			ce += -target[i] * log(o);
		}
		return ce / k;
	}
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {
	Layer &outputLayer = layers[nOfLayers -1];
	int k = outputLayer.nOfNeurons;

	// Output layer deltas
	for (int i = 0; i < k; i++)
	{
		double out = outputLayer.neurons[i].out;
		if (errorFunction == 1 && outputFunction == 1) {
			// Cross-entropy + softmax: delta = (o - d)
			outputLayer.neurons[i].delta = out - target[i];
		} else {
			// MSE + sigmoid (default)
			double error = target[i] - out;
			outputLayer.neurons[i].delta = error * out * (1.0 - out);
		}
	}

	// Hidden layers
	for (int i = nOfLayers - 2; i > 0; i--)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double out = layers[i].neurons[j].out;
			double sum = 0.0;

			for (int k2 = 0; k2 < layers[i+1].nOfNeurons; k2++)
			{
				sum += layers[i+1].neurons[k2].delta * layers[i+1].neurons[k2].w[j];
			}
			layers[i].neurons[j].delta = out * (1.0 - out) * sum;
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for (int i = 1; i < nOfLayers; i++) {
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			Neuron &neuron = layers[i].neurons[j];
			int nInputs = layers[i-1].nOfNeurons;
			for (int k = 0; k < nInputs; k++) {
				double outPrev = layers[i-1].neurons[k].out;
				neuron.deltaW[k] += neuron.delta * outPrev;
			}
			// bias
			neuron.deltaW[nInputs] += neuron.delta * 1.0;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	for (int i = 1; i < nOfLayers; i++) { // skip input layer
		for (int j = 0; j < layers[i].nOfNeurons; j++) {
			Neuron& neuron = layers[i].neurons[j];
			int nInputs = layers[i - 1].nOfNeurons;

			for (int k = 0; k < nInputs; k++) {
				double gradTerm;
				if (online) {
					double outPrev = layers[i - 1].neurons[k].out;
					gradTerm = neuron.delta * outPrev;
				} else {
					gradTerm = neuron.deltaW[k] / (double)nOfTrainingPatterns; // average (batch)
				}
				double newDeltaW = eta * gradTerm + mu * neuron.lastDeltaW[k];
				neuron.w[k] += newDeltaW;

				neuron.lastDeltaW[k] = newDeltaW;
				if (!online) neuron.deltaW[k] = 0.0; // clear accumulator
			}

			// bias weight
			double gradBias = online ? (neuron.delta * 1.0)
			                         : (neuron.deltaW[nInputs] / (double)nOfTrainingPatterns);
			double newDeltaWbias = eta * gradBias + mu * neuron.lastDeltaW[nInputs];
			neuron.w[nInputs] += newDeltaWbias;
			neuron.lastDeltaW[nInputs] = newDeltaWbias;

			if (!online) neuron.deltaW[nInputs] = 0.0;
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


// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	if (online) {
		weightAdjustment();
	} else {
		accumulateChange();
	}
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	// Reset accumulators for offline mode
	if (!online) {
		for (int i = 1; i < nOfLayers; i++) {
			for (int j = 0; j < layers[i].nOfNeurons; j++) {
				int nInputs = layers[i-1].nOfNeurons + 1;
				for (int k = 0; k < nInputs; k++)
					layers[i].neurons[j].deltaW[k] = 0.0;
			}
		}
	}

	// Order of patterns: shuffle for online (assignment hint), sequential for offline
	if (online) {
		int *idx = util::integerRandomVectoWithoutRepeating(0, trainDataset->nOfPatterns - 1, trainDataset->nOfPatterns);
		for (int p = 0; p < trainDataset->nOfPatterns; p++) {
			int t = idx[p];
			performEpoch(trainDataset->inputs[t], trainDataset->outputs[t], errorFunction);
		}
		delete[] idx;
	} else {
		for (int p = 0; p < trainDataset->nOfPatterns; p++) {
			performEpoch(trainDataset->inputs[p], trainDataset->outputs[p], errorFunction);
		}
		weightAdjustment(); // apply averaged accumulated changes once per outer iteration
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	double sumErr = 0.0;
	for (int p = 0; p < dataset->nOfPatterns; p++) {
		feedInputs(dataset->inputs[p]);
		forwardPropagate();
		sumErr += obtainError(dataset->outputs[p], errorFunction);
	}
	return sumErr / dataset->nOfPatterns;
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
	int correct = 0;
	int k = layers[nOfLayers - 1].nOfNeurons;

	for (int p = 0; p < dataset->nOfPatterns; p++) {
		feedInputs(dataset->inputs[p]);
		forwardPropagate();

		int y_true = 0, y_pred = 0;
		for (int i = 1; i < k; i++) {
			if (dataset->outputs[p][i] > dataset->outputs[p][y_true]) y_true = i;
		}
		for (int i = 1; i < k; i++) {
			if (layers[nOfLayers - 1].neurons[i].out > layers[nOfLayers - 1].neurons[y_pred].out) y_pred = i;
		}
		if (y_true == y_pred) correct++;
	}
	return 100.0 * ((double)correct / (double)dataset->nOfPatterns);
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

    double minTrainError = 0.0;
    int iterWithoutImproving = 0;
    const double tol = 1e-5;
    nOfTrainingPatterns = trainDataset->nOfPatterns;

    // Learning
    do {
        train(trainDataset, errorFunction);

        double trainError = test(trainDataset, errorFunction);

        if (countTrain == 0 || trainError < minTrainError - tol) {
            minTrainError = trainError;
            copyWeights();
            iterWithoutImproving = 0;
        } else {
            // improvement <= tol counts as "no improvement"
            iterWithoutImproving++;
        }

        if (iterWithoutImproving >= 50) {
            cout << "We exit because the training is not improving!!" << endl;
            break;
        }

        countTrain++;
        cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;
    } while (countTrain < maxiter);

    // Restore best weights before evaluation
    restoreWeights();

    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    printNetwork();

    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;
    for(int i=0; i<testDataset->nOfPatterns; i++){
        double* prediction = new double[testDataset->nOfOutputs];

        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        for(int j=0; j<testDataset->nOfOutputs; j++)
            cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
        delete[] prediction;
    }

    *errorTest = test(testDataset, errorFunction);
    *errorTrain = minTrainError;
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
