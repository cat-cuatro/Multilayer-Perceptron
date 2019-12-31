#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <time.h>
#define NUM_INPUTS 784
#define NUM_OUTPUTS 10
#define BIAS 1
#define LEARNING_RATE .001
#define TRAIN true
#define TEST false
#define EPOCHS 70
#define CELLS 10
#define TEST_ENTRIES 10000
#define TRAIN_ENTRIES 60000
#define HIDDEN_UNITS 20
#define HIDDEN_OUTPUTS 10
#define BATCH_SIZE 100
#define MOMENTUM 0.25
using namespace std;

class perceptron{
  public:
    perceptron();
    ~perceptron();
    /*** Functions ***/
    double randomWeight();
    double computeOutput(int maxsize);
    void assignInputs(perceptron hidden_net[], perceptron output_net[]);
    double firing();
    double notFiring();
    void printInputs();
    void forwardFeed();
    void prog1Learn(double prediction, double target, int maxsize);
    double calcErrorOutput(double target, double an_output);
    double calcErrorHidden(perceptron output_net[], double sumErr, double weightsArr[], int index);
    double sigmoid(double an_output);
    bool assertPrediction(double prediction, double target);
    void initHiddenLayer();
    void updateHiddenToOutput(perceptron output_net[], double outputChanges[], double sumError);
    void updateInputToHidden(perceptron hidden_net[], double inputChanges[], int index);
    /*** Variables ***/
    class perceptron * hidden;
    double inputs[NUM_INPUTS];
    double bias;
    double error; // error for single outputs
    double bias_weight;
    double weights[NUM_INPUTS]; // weights for regular inputs
//    double weightErrors[HIDDEN_UNITS]; // weight errors for weights connected to output net
    double output;
    double outputWeights[10];
    double lastChange[10];
    double hiddenErrors[10]; // hidden layer sends it output to 10 different output cells
  private:                               // which means each output cell has HIDDEN_UNITS number of inputs
                                         // Therefore, there are 20 different errors 
};
double pla(perceptron simple_net[], perceptron output_net[], bool training, int confusionMatrix[][10]);
bool assertPrediction(double prediction, double target);
void learn(perceptron simple_net[], double predic, double targ);
void printNum(perceptron simple_net[]);
void initWeights(perceptron simple_net[], int maxsize);
void printMatrix(int confusionMatrix[][10]);
void initMatrix(int confusionMatrix[][10]);
void incrementMatrix(int confusionMatrix[][10], double predic, double targ);
void initArrays(double changes[], int maxsize);
