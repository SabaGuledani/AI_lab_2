//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, lflag = 0, eflag = 0, mflag = 0, iflag = 0, hflag = 0, sflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL;
    int c;

     // Training parameters
    double eta = 0.1;     // Learning rate
    double mu = 0.9;      // Momentum
    int iterations = 1000;
    int layers = 0;          // number of hidden layers
    int hiddenNeurons = 5;   // neurons per hidden layer
    int *topology = NULL;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:w:l:h:e:m:i:ps")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
         switch(c){
            case 't':   // Training dataset
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':   // Test dataset
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'w':   // Weights file
                wflag = true;
                wvalue = optarg;
                break;
            case 'l':   // Number of hidden layers
                lflag = true;
                layers = atoi(optarg);
                break;
            case 'h':   // Neurons per hidden layer
                hflag = true;
                hiddenNeurons = atoi(optarg);
                break;
            case 'e':   // Learning rate
                eflag = true;
                eta = atof(optarg);
                break;
            case 'm':   // Momentum
                mflag = true;
                mu = atof(optarg);
                break;
            case 'i':   // Iterations
                iflag = true;
                iterations = atoi(optarg);
                break;
            case 'p':   // Prediction mode
                pflag = true;
                break;
            case 's':   // Normalize datasets after reading
                sflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 't' || optopt == 'l' || optopt == 'h' ||
                    optopt == 'e' || optopt == 'm' || optopt == 'i')
                    fprintf(stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr, "Unknown character `\\x%x'.\n", optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        if (!tflag || !lflag) {
            cerr << "Usage: ./la1 -t train.txt [-T test.txt] -l nHiddenLayers [-h hiddenNeurons] [-i iter -e eta -m mu -s -w weights]" << endl;
            return EXIT_FAILURE;
        }

        // Multilayer perceptron object
		MultilayerPerceptron mlp;
        mlp.eta = eta;
        mlp.mu = mu;

        // Read training and test data
		Dataset * trainDataset = readData(tvalue);
        if(trainDataset == NULL){
            cerr << "The training file is not valid, we can not continue" << endl;
            return EXIT_FAILURE;
        }
		Dataset * testDataset = NULL;
        if (Tflag) {
            testDataset = readData(Tvalue);
            if(testDataset == NULL){
                cerr << "The test file is not valid, we can not continue" << endl;
                return EXIT_FAILURE;
            }
        } else {
            testDataset = trainDataset;
        }

        // Apply normalization if requested: inputs to [-1,1], outputs to [0,1]
        double *minInputs = NULL; double *maxInputs = NULL; double minOut = 0.0; double maxOut = 0.0;
        if (sflag) {
            minInputs = minDatasetInputs(trainDataset);
            maxInputs = maxDatasetInputs(trainDataset);
            minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputs, maxInputs);
            if (testDataset != trainDataset)
                minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputs, maxInputs);

            // Outputs (especially for regression)
            minOut = minDatasetOutputs(trainDataset);
            maxOut = maxDatasetOutputs(trainDataset);
            minMaxScalerDataSetOutputs(trainDataset, 0.0, 1.0, minOut, maxOut);
            if (testDataset != trainDataset)
                minMaxScalerDataSetOutputs(testDataset, 0.0, 1.0, minOut, maxOut);
        }

        // Build topology: [inputs, hidden x L, outputs]
        topology = new int[layers + 2];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 1; i <= layers; ++i) topology[i] = hiddenNeurons;
        topology[layers + 1] = trainDataset->nOfOutputs;

        // Initialize topology vector
        //int *topology = new int[layers+2];
        //topology[0] = trainDataset->nOfInputs;
        //for(int i=1; i<(layers+2-1); i++)
        //    topology[i] = neurons;
        //topology[layers+2-1] = trainDataset->nOfOutputs;
        //mlp.initialize(layers+2,topology);

        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // Obtain training and test averages and standard deviations
        for (int i = 0; i < 5; ++i) {
            averageTrainError += trainErrors[i];
            averageTestError += testErrors[i];
        }
        averageTrainError /= 5.0;
        averageTestError /= 5.0;

        double varTrain = 0.0, varTest = 0.0;
        for (int i = 0; i < 5; ++i) {
            double dt = trainErrors[i] - averageTrainError;
            double dv = testErrors[i] - averageTestError;
            varTrain += dt * dt;
            varTest += dv * dv;
        }
        stdTrainError = sqrt(varTrain / 5.0);
        stdTestError = sqrt(varTest / 5.0);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        // Cleanup
        delete[] testErrors;
        delete[] trainErrors;
        delete[] topology;
        if (sflag) { delete[] minInputs; delete[] maxInputs; }

        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;

	}
}

