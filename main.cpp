#include "neural.h"

int main(){
  perceptron simple_net[NUM_OUTPUTS];
  perceptron hidden_net[HIDDEN_UNITS];
  initWeights(simple_net, NUM_OUTPUTS);
  initWeights(hidden_net, HIDDEN_UNITS);
  double success_percent = 0;
  int confusionMatrix[10][10];
  double catching = 0;
  initMatrix(confusionMatrix);
  cout << "weight: " << simple_net[5].weights[5] << endl;
  cout << "bias: " << simple_net[5].bias << " // bias weight: " << simple_net[5].bias_weight<< endl;
  for(int i = 0; i < 5000; ++i){
    catching = simple_net[0].randomWeight();
    if(catching > .05 || catching < -.05){
      cout << "Bad! " << catching << endl;
    }
  }
  for(int i = 0; i < EPOCHS; ++i){
    cout << "TRAINING the perceptron now . . ." << endl;
    success_percent = pla(simple_net, hidden_net, TRAIN, confusionMatrix);
    cout << "-- End of Epoch: " << i << ", Accuracy: " << success_percent << "% --" << endl;
//    cout << "TESTING the perceptron now . . ." << endl;
//    success_percent = pla(simple_net, hidden_net, TEST, confusionMatrix);
//    cout << "-- End of Epoch: " << i << ", Accuracy: " << success_percent << "% --" << endl;
  }
  
  return 0;
}
double pla(perceptron simple_net[], perceptron output_net[], bool training, int confusionMatrix[][10]){
  int i,j,k,n,m,z;
  double prediction = 0; // prediction label
  double target = 0; // target label
  double highestOutput = 0;
  double current = 0;
  int targetint = 0;
  int entries = 0;
  ifstream mnist_data;
  int correct = 0;
  int incorrect = 0;
  time_t start = time(0);
  double seconds_since_start = 0;
  double success_percent = 0;
  double outLayerErrors[NUM_OUTPUTS];
  double outputChanges[10];
  double inputChanges[HIDDEN_UNITS];
  double sumError = 0;
  initArrays(outputChanges, 10);
  initArrays(inputChanges, HIDDEN_UNITS);
//  double outputErrors[HIDDEN_UNITS];
    // 1. Feed data to each cell in the net
    // 2. Compute each of their outputs
    // 3. Compare the weights. The highest weight of all 10 cells is the number prediction
    // 4. If wrong, then train the weights.
    // 5. Run again for some amount of epochs
  if(training){
    entries = TRAIN_ENTRIES;
  }
  else{
    entries = TEST_ENTRIES;
  }
  for(j = 0; j < 1; ++j){ // run this algorithm many times
    correct = 0;
    incorrect = 0;
    if(training){ // If training, then open training data
      mnist_data.open("mnist_train.csv");
    }
    else{         // if testing, then open testing data
      mnist_data.open("mnist_test.csv");
    }
    for(n = 0; n < entries; ++n){ // for the number of entries in the data
      mnist_data >> target; // grab target label
      mnist_data.ignore(100, ',');
      for(k = 0; k < NUM_INPUTS; ++k){
        mnist_data >> simple_net[0].inputs[k]; // read MNIST data into the inputs
        if(k != NUM_INPUTS-1){
          mnist_data.ignore(100, ',');
        }
        simple_net[0].inputs[k] = simple_net[0].inputs[k]/255.0; // normalize data
        simple_net[1].inputs[k] = simple_net[0].inputs[k];     // each cell will receive the same input
        simple_net[2].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[3].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[4].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[5].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[6].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[7].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[8].inputs[k] = simple_net[0].inputs[k]; 
        simple_net[9].inputs[k] = simple_net[0].inputs[k];
        // 'unrolling' to maybe save some computation time
      }
      mnist_data.ignore(100, '\n'); // ignore newline

      if(n % 10000 == 0){
//        cout << "Target: " << target << endl;
//        printNum(simple_net);
      }
      for(i = 0; i < HIDDEN_UNITS; ++i){ // first compute the outputs of the hidden layer
        simple_net[i].output = simple_net[i].computeOutput(NUM_INPUTS);
      }
      /*** Now I need to assign the outputs of the hidden layer to the inputs of the output layer ***/
      output_net[0].assignInputs(simple_net, output_net); // <-- That is accomplished here
      
      for(i = 0; i < NUM_OUTPUTS; ++i){ // then compute the outputs of the output layer
        output_net[i].output = output_net[i].computeOutput(HIDDEN_UNITS);
        current = output_net[i].output;
//        output_net[i].error = calcErrorOutput(target, current);        
        /*** Now make a numerical prediction based on outputs ***/
        if(highestOutput < current){
          highestOutput = current;
          prediction = i; // set the number prediction
        }
      }

      if(n%10000 == 0){
        cout << "Target is: " << target << endl;
        cout << "Prediction is: " << prediction << endl;
      }
      if(n == 0){
        cout << "output[0].weights[1]: " << output_net[0].weights[1] << endl;
        cout << "hidden[0].outputWeights[1]: " << simple_net[0].outputWeights[1] << endl;
      }

      if(!assertPrediction(prediction, target)){ // if false prediction and we're training, then adjust weights
      // train algorithm here
        if(training){ // replace with gradient descent learning
//          learn(simple_net, prediction, target); 
          for(i = 0; i < NUM_OUTPUTS; ++i){ // <-- Find the neuron-specific errors
            if(i == prediction){ 
              output_net[i].error = output_net[i].calcErrorOutput(0.9, output_net[i].output); 
              outLayerErrors[i] = output_net[i].error; // store errors locally
              sumError = sumError + output_net[i].error;
              // Target is a 1 because in a vector that represents a number, it has nine 0s and one 1 to represent a number.
              // Ex: <0, 0, 0, 1, 0 . . .> = 3 -- another example: <0, 1, 0, 0 . . .> = 1
            }
            else{
              output_net[i].error = output_net[i].calcErrorOutput(0.1, output_net[i].output);
              outLayerErrors[i] = output_net[i].error; // store errors locally
              sumError = sumError + output_net[i].error;
            }
          }
          for(i = 0; i < 10; ++i){ // for each cell's error (10 of them)
            for(m = 0; m < HIDDEN_UNITS; ++m){ // for each hidden cell (20 of them)
              simple_net[m].error = simple_net[m].calcErrorHidden(output_net, output_net[i].error, output_net[i].weights, m); 
            }
          }
          for(i = 0; i < 10; ++i){
            simple_net[0].updateHiddenToOutput(output_net, outputChanges, output_net[i].error);
          }
          for(i = 0; i < HIDDEN_UNITS; ++i){
            simple_net[i].updateInputToHidden(simple_net, inputChanges, i);
          }
        }
        ++incorrect;
      }
      if(training == true){
        incrementMatrix(confusionMatrix, prediction, target); // update confusion matrix
      }
      highestOutput = 0;
      prediction = 0;
    }
    mnist_data.close();
    success_percent = (((double)entries-incorrect)/(double)entries)*100.0;
//    cout << "-- End of Epoch: " << j << ", Incorrect: " << incorrect << " times. Accuracy: " << success_percent << "% --" << endl;
    cout << "Time elapsed: " << difftime(time(0), start) << " seconds." << " Incorrect: " << incorrect << " times." << endl;
    
    if(training == true){
      printMatrix(confusionMatrix);
    }
  }
  return success_percent;
}
