#include "neural.h"
/*** Perceptron class methods ***/
// Create and initialize all perceptron values.
perceptron::perceptron(){ 
/*
  for(int i = 0; i < NUM_INPUTS; ++i){
    weights[NUM_INPUTS] = randomWeight();
    cout << "const. weight: " << weights[i] << endl;
  }
*/
  bias = 1;
  output = 0;
  error = 0;
  bias_weight = randomWeight();
//  cout << "Constructor!" << endl;
}
// End of lifetime, therefore destroy data
perceptron::~perceptron(){
  delete [] hidden;
}
// Computes a weight between -.05 and .05. Intnetionally pseudo-random
double perceptron::randomWeight(){
  double weight = -0.05;
  double add = ((double) rand() / (RAND_MAX));
  add = add/10;
  weight = weight + add;
  return weight;
}
void perceptron::printInputs(){
  for(int i = 0; i < NUM_INPUTS; ++i){
    cout << inputs[i] << ",";
  }
  cout << endl;
}
// Computes output for this neuron
double perceptron::computeOutput(int maxsize){
  double localOutput = 0;
  for(int i = 0; i < maxsize; ++i){
    localOutput = (inputs[i] * weights[i]) + localOutput;
  }
  localOutput = bias * bias_weight + localOutput;
  // sigmoid??
  localOutput = sigmoid(localOutput);
  return localOutput;
}
double perceptron::firing(){
  return 1;
}
double perceptron::notFiring(){
  return 0;
}
void perceptron::initHiddenLayer(){ // initialize the hidden layer
  int i = 0;
  int j = 0;
  hidden = new perceptron[HIDDEN_UNITS];
  for(i = 0; i < HIDDEN_UNITS; ++i){
    hidden[i].bias = 1;
    hidden[i].bias_weight = randomWeight();
    hidden[i].output = 0;
    for(j = 0; j < NUM_INPUTS; ++j){
      hidden[i].inputs[j] = 0;
      hidden[i].weights[j] = randomWeight();
    }
  }
//  cout << "Initialized " << HIDDEN_UNITS << " hidden layers." << endl;
}
double perceptron::sigmoid(double an_output){
  double activation = 0;
  activation = (1/(1+exp(an_output)));
  return activation;
}
void perceptron::assignInputs(perceptron hidden_net[], perceptron output_net[]){
  int i = 0;
  int j = 0;
  int hidden_output = 0;
  for(i = 0; i < HIDDEN_UNITS; ++i){
//    hidden_output = hidden_net[i].output;
    for(j = 0; j < 10; ++j){
      output_net[j].inputs[i] = hidden_net[i].output;
      hidden_net[i].outputWeights[j] = output_net[j].weights[i];
/*
      output_net[0].inputs[i] = hidden_output;
      output_net[1].inputs[i] = hidden_output;
      output_net[2].inputs[i] = hidden_output;
      output_net[3].inputs[i] = hidden_output;
      output_net[4].inputs[i] = hidden_output;
      output_net[5].inputs[i] = hidden_output;
      output_net[6].inputs[i] = hidden_output;
      output_net[7].inputs[i] = hidden_output;
      output_net[8].inputs[i] = hidden_output;
      output_net[9].inputs[i] = hidden_output; 
*/
    }
  }
}
// Uses the target label and this neuron's output to calculate error
double perceptron::calcErrorOutput(double target, double an_output){
  double err = 0; 
  err = an_output*(1-an_output)*(an_output-target);
  return err;
}
// Uses it's parent neuron's output, it's own output, and the weight for this input to find error
double perceptron::calcErrorHidden(perceptron output_net[], double sumErr, double weightsArr[], int index){
  double err = 0;
  double summed = 0;
  int i,j,k;
//  for(i = 0; i < 10; ++i){
//    hiddenErrors[i] = output*(1-output)*outputWeights[i]*output_net[i].error;
   // err = output*(1-output)*outputWeights[i]*output_net[i].error;
//    err = output*(1-output)*outputWeights[i]*(sumErr) + err;
      err = output*(1-output)*weightsArr[index]*sumErr;
//    cout << err << endl;
      hiddenErrors[index] = err;
//  }
  //err = an_output*(1-an_output)*a_weight*outputError;
  return err;
}
void perceptron::updateHiddenToOutput(perceptron output_net[], double outputChanges[], double sumError){
  double change = 0;
  int i,j;
  for(i = 0; i < 10; ++i){
    for(j = 0; j < HIDDEN_UNITS; ++j){
      if(i == 4){
//        cout << "Weight before: " << output_net[i].weights[j] << endl;
      }
     // change = LEARNING_RATE*output_net[i].error*output_net[i].inputs[j] + MOMENTUM*outputChanges[i];
      // weight = learn_rate*output_error*input + momentum*lastchange
      change = LEARNING_RATE*(sumError)*output_net[i].inputs[j] + MOMENTUM*outputChanges[i];
      output_net[i].weights[j] = output_net[i].weights[j] + change;
      if(i == 4){
//        cout << "Weight after: " << output_net[i].weights[j] << endl;
      }
    }
  }
}
void perceptron::updateInputToHidden(perceptron hidden_net[], double inputChanges[], int index){
  double change = 0;
  int i,j;
  for(i = 0; i < HIDDEN_UNITS; ++i){
    for(j = 0; j < NUM_INPUTS; ++j){
//      cout << "Weight before: " << hidden_net[i].weights[j] << endl;
//      change = LEARNING_RATE*hidden_net[i].hiddenErrors[i]*hidden_net[i].inputs[j] + MOMENTUM*inputChanges[i];
      change = LEARNING_RATE*hiddenErrors[i]*hidden_net[i].inputs[j] + MOMENTUM*inputChanges[i];
      hidden_net[i].weights[j] = hidden_net[i].weights[j] + change;
//      cout << "Weight after: " << hidden_net[i].weights[j] << endl;
    }
  }
}
/*******************************************************************************************/
/***                              Class-less methods                                     ***/
/***                                                                                     ***/

// Check if the perceptron was right or wrong
bool assertPrediction(double prediction, double target){
  if(prediction == target){
    return true;
  }
  else{
    return false;
  }
}
// Train the weights in the perceptron
void learn(perceptron simple_net[], double predic, double targ){
  int i,j,k;
  for(i = 0; i < CELLS; ++i){ // for each cell
    for(j = 0; j < NUM_INPUTS; ++j){ // for each weight in the cell
      if(i == targ){
        simple_net[i].weights[j] = simple_net[i].weights[j] + LEARNING_RATE*(1-0)*simple_net[i].inputs[j];
      }
      else{
        simple_net[i].weights[j] = simple_net[i].weights[j] + LEARNING_RATE*(0 - 0)*simple_net[i].inputs[j];
      } 
    }
    simple_net[i].bias_weight = simple_net[i].bias_weight + LEARNING_RATE*(1-0)*simple_net[i].bias;
  }
}
void progLearn(double prediction, double target, int maxsize){
  int i,j,k;
  for(i = 0; i < maxsize; ++i){
    
  }
}
void printNum(perceptron simple_net[]){
  int i,j,k;
  int internalCount = 0;
  for(i = 0; i < NUM_INPUTS; ++i){
    if(internalCount < 28){
      if(simple_net[5].inputs[i] > 0){
        cout << "X";
      }
      else{
        cout << ".";
      }
    }
    else{
     if(simple_net[5].inputs[i] > 0){
        cout << "X";
      }
      else{
        cout << ".";
      } 
      cout << endl;
      internalCount = 0;
    }
    ++internalCount;
  }
  cout << endl;
}
void initWeights(perceptron simple_net[], int maxsize){
  int i,j;
  for(i = 0; i < maxsize; ++i){
    for(j = 0; j < NUM_INPUTS; ++j){
      simple_net[i].weights[j] = simple_net[i].randomWeight();
//      simple_net[i].weightErrors[j] = 0;
    }
//    simple_net[i].initHiddenLayer(); // not needed
    simple_net[i].bias = 1;
    simple_net[i].output = 0;
  }
}
void printMatrix(int confusionMatrix[][10]){
  int i,j;
  cout << "X: \t0\t1\t2\t3\t4\t5\t6\t7\t8\t9" << endl;
  for(i = 0; i < 10; ++i){
    cout << i << ":";
    for(j = 0; j < 10; ++j){
      cout << "\t" << confusionMatrix[i][j];
    }
    cout << endl;
  }
}
void initMatrix(int confusionMatrix[][10]){
  int i,j;
  for(i = 0; i < 10; ++i){
    for(j = 0; j < 10; ++j){
      confusionMatrix[i][j] = 0;
    }
  }
}
void incrementMatrix(int confusionMatrix[][10], double predic, double targ){
  int x = (int)predic;
  int y = (int)targ;
  ++confusionMatrix[x][y];
}
void initArrays(double changes[], int maxsize){
  int i = 0;
  for(i = 0; i < maxsize; ++i){
    changes[i] = 0;
  }
}
