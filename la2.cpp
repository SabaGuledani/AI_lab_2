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
    // process command line arguments
    bool Tflag = false, wflag = false, pflag = false, tflag = false;
    bool lflag = false, eflag = false, mflag = false, iflag = false, hflag = false;
    bool nflag = false, oflag = false, sflag = false, fflag = false;

    double eta = 0.7;
    double mu = 1.0;
    int iterations = 1000;
    int layers = 1;              // default is 1 hidden layer (for part 1 & 2)
    int hiddenNeurons = 5;
    int errorFunction = 0;       // 0=MSE, 1=Cross-Entropy

    bool normalizeData = false;  // normalize inputs if needed
    bool onlineMode = false;     // offline by default
    bool useSoftmax = false;     // sigmoid by default

    // getopt state and values
    int c;
    const char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;
    int *topology = NULL;

    opterr = 0;


     // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:w:l:h:e:m:i:psof:n")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
         switch(c){
            case 't': tflag = true; tvalue = optarg; break;
            case 'T': Tflag = true; Tvalue = optarg; break;
            case 'w': wflag = true; wvalue = optarg; break;
            case 'l': lflag = true; layers = atoi(optarg); break;
            case 'h': hflag = true; hiddenNeurons = atoi(optarg); break;
            case 'e': eflag = true; eta = atof(optarg); break;
            case 'm': mflag = true; mu = atof(optarg); break;
            case 'i': iflag = true; iterations = atoi(optarg); break;
            case 'p': pflag = true; break;
            case 's': sflag = true; useSoftmax = true; break;     // enable softmax
            case 'o': oflag = true; onlineMode = true; break;     // enable online mode (offline is default)
            case 'f': fflag = true; errorFunction = atoi(optarg); break;
            case 'n': nflag = true; normalizeData = true; break;  // normalize the inputs
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        if (!tflag || !lflag) {
            cerr << "Usage: ./la2 -t train.txt [-T test.txt] -l nHiddenLayers [-h hiddenNeurons] [-i iter -e eta -m mu -o -f {0|1} -s -n -w weights]" << endl;
            return EXIT_FAILURE;
        }

        // create multilayer perceptron object
		MultilayerPerceptron mlp;
        mlp.eta = eta;
        mlp.mu = mu;
        mlp.online = onlineMode;
        mlp.outputFunction = useSoftmax ? 1 : 0;

        // read training and test data
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

        // Apply normalization if requested: inputs to [-1,1] (outputs are NOT normalized in classification)
        double *minInputs = NULL; double *maxInputs = NULL;
        if (nflag) {
            minInputs = minDatasetInputs(trainDataset);
            maxInputs = maxDatasetInputs(trainDataset);
            minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputs, maxInputs);
            if (testDataset != trainDataset)
                minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputs, maxInputs);
        }

        // build topology: [inputs, hidden x L, outputs]
        topology = new int[layers + 2];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 1; i <= layers; ++i) topology[i] = hiddenNeurons;
        topology[layers + 1] = trainDataset->nOfOutputs;

        if (!mlp.initialize(layers + 2, topology)) {
            cerr << "Error initializing network topology." << endl;
            return EXIT_FAILURE;
        }

        // seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double *testCCR = new double[5];
        double *trainCCR = new double[5];
        double bestTestError = DBL_MAX;
        for (int i = 0; i < 5; ++i) {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
        
            double errTrain = 0.0, errTest = 0.0, ccrTrain = 0.0, ccrTest = 0.0;
            mlp.runBackPropagation(
                trainDataset,
                testDataset,
                iterations,
                &errTrain,
                &errTest,
                &ccrTrain,
                &ccrTest,
                errorFunction
            );
        
            trainErrors[i] = errTrain;
            testErrors[i] = errTest;
            trainCCR[i] = ccrTrain;
            testCCR[i] = ccrTest;
        
            cout << "We end!! => Final test error: " << testErrors[i]
                 << " | Final test CCR: " << testCCR[i] << endl;
        
            if (wflag && testErrors[i] <= bestTestError) {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        double averageTestCCR = 0, stdTestCCR = 0;
        double averageTrainCCR = 0, stdTrainCCR = 0;

        // calculate training and test averages and standard deviations
        for (int i = 0; i < 5; ++i) {
            averageTrainError += trainErrors[i];
            averageTestError += testErrors[i];
            averageTrainCCR += trainCCR[i];
            averageTestCCR += testCCR[i];
        }
        averageTrainError /= 5.0;
        averageTestError /= 5.0;
        averageTrainCCR /= 5.0;
        averageTestCCR /= 5.0;

        double varTrain = 0.0, varTest = 0.0, varTrainCCR = 0.0, varTestCCR = 0.0;
        for (int i = 0; i < 5; ++i) {
            double dt = trainErrors[i] - averageTrainError;
            double dv = testErrors[i] - averageTestError;
            varTrain += dt * dt;
            varTest += dv * dv;

            double dct = trainCCR[i] - averageTrainCCR;
            double dcv = testCCR[i] - averageTestCCR;
            varTrainCCR += dct * dct;
            varTestCCR += dcv * dcv;
        }
        stdTrainError = sqrt(varTrain / 5.0);
        stdTestError = sqrt(varTest / 5.0);
        stdTrainCCR = sqrt(varTrainCCR / 5.0);
        stdTestCCR = sqrt(varTestCCR / 5.0);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error  (Mean +- SD): " << averageTestError  << " +- " << stdTestError  << endl;
        cout << "Train CCR   (Mean +- SD): " << averageTrainCCR   << " +- " << stdTrainCCR   << endl;
        cout << "Test CCR    (Mean +- SD): " << averageTestCCR    << " +- " << stdTestCCR    << endl;

        // Cleanup
        delete[] testErrors;
        delete[] trainErrors;
        delete[] testCCR;
        delete[] trainCCR;
        delete[] topology;
        if (nflag) { delete[] minInputs; delete[] maxInputs; }

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

