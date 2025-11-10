#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <vector>
#include <algorithm>

#include "util.h"

using namespace std;
using namespace util;


// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
	return rand() % (High-Low+1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
	return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

// ------------------------------
// read a dataset from a file name and return it
Dataset *util::readData(const char *fileName)
{

    ifstream myFile(fileName); // create an input stream

    if (!myFile.is_open())
    {
        cout << "ERROR: I cannot open the file " << fileName << endl;
        return NULL;
    }

    Dataset *dataset = new Dataset;
    if (dataset == NULL)
        return NULL;

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // read a line
        istringstream iss(line);
        iss >> dataset->nOfInputs;
        iss >> dataset->nOfOutputs;
        iss >> dataset->nOfPatterns;
    }
    dataset->inputs = new double *[dataset->nOfPatterns];
    dataset->outputs = new double *[dataset->nOfPatterns];

    for (i = 0; i < dataset->nOfPatterns; i++)
    {
        dataset->inputs[i] = new double[dataset->nOfInputs];
        dataset->outputs[i] = new double[dataset->nOfOutputs];
    }

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset->nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->inputs[i][j] = value;
            }
            for (j = 0; j < dataset->nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}


// ------------------------------
// Print the dataset
void util::printDataset(Dataset *dataset, int len)
{
    if (len == 0)
        len = dataset->nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset->nOfInputs; j++)
        {
            cout << dataset->inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            cout << dataset->outputs[i][j] << ",";
        }
        cout << endl;
    }
}

// ------------------------------
// transform a scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData)
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    if (maxData == minData) {
        // avoid division by zero: map to midpoint of allowed range
        return (minAllowed + maxAllowed) / 2.0;
    }
    double scaled = (x - minData) / (maxData - minData);            // [0,1]
    return minAllowed + scaled * (maxAllowed - minAllowed);         // [minAllowed,maxAllowed]
}

// ------------------------------
// scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData)
void util::minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                     double *minData, double *maxData)
{
    // scale each feature (column) independently using provided min/max vectors
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfInputs; ++j)
        {
            dataset->inputs[p][j] = minMaxScaler(dataset->inputs[p][j], minAllowed, maxAllowed,
                                                 minData[j], maxData[j]);
        }
    }
}

// ------------------------------
// Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). Only for regression problems
void util::minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfOutputs; ++j)
        {
            dataset->outputs[p][j] = minMaxScaler(dataset->outputs[p][j], minAllowed, maxAllowed,
                                                  minData, maxData);
        }
    }
}

// ------------------------------
// Get a vector of minimum values of the dataset inputs
double *util::minDatasetInputs(Dataset *dataset)
{
    double *mins = new double[dataset->nOfInputs];
    for (int j = 0; j < dataset->nOfInputs; ++j)
    {
        mins[j] = dataset->inputs[0][j];
    }
    for (int p = 1; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfInputs; ++j)
        {
            if (dataset->inputs[p][j] < mins[j]) mins[j] = dataset->inputs[p][j];
        }
    }
    return mins;
}

// ------------------------------
// Get a vector of maximum values of the dataset inputs
double *util::maxDatasetInputs(Dataset *dataset)
{
    double *maxs = new double[dataset->nOfInputs];
    for (int j = 0; j < dataset->nOfInputs; ++j)
    {
        maxs[j] = dataset->inputs[0][j];
    }
    for (int p = 1; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfInputs; ++j)
        {
            if (dataset->inputs[p][j] > maxs[j]) maxs[j] = dataset->inputs[p][j];
        }
    }
    return maxs;
}

// ------------------------------
// Get the minimum value of the dataset outputs
double util::minDatasetOutputs(Dataset *dataset)
{
    double m = dataset->outputs[0][0];
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfOutputs; ++j)
        {
            if (dataset->outputs[p][j] < m) m = dataset->outputs[p][j];
        }
    }
    return m;
}

// ------------------------------
 // get the maximum value of the dataset outputs
double util::maxDatasetOutputs(Dataset *dataset)
{
    double m = dataset->outputs[0][0];
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int j = 0; j < dataset->nOfOutputs; ++j)
        {
            if (dataset->outputs[p][j] > m) m = dataset->outputs[p][j];
        }
    }
    return m;
}

int* util::integerRandomVectoWithoutRepeating(int low, int high, int n)
{
    if (high - low + 1 != n) {
        // fallback: build based on n
        low = 0;
        high = n - 1;
    }
    std::vector<int> v;
    v.reserve(n);
    for (int x = low; x <= high; ++x) v.push_back(x);
    // fisher-yates shuffle using rand()
    for (int i = n - 1; i > 0; --i) {
        int j = util::randomInt(0, i);
        std::swap(v[i], v[j]);
    }
    int* out = new int[n];
    for (int i = 0; i < n; ++i) out[i] = v[i];
    return out;
}

