#include "string"
#include <iostream>

#include <algorithm>
#include <core/model.h>
#include <fstream>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  //       and saves the trained model to a file.

  naivebayes::Model model;

  std::ifstream training_image_stream("../data/trainingimagesandlabels.txt");

  training_image_stream >> model;

  model.Train();

  model.Save("../saved/", "saved_model.txt");
  model.Load("../saved/saved_model.txt");
  model.Save("../saved/", "saved_model_after.txt");

  return 0;
}
