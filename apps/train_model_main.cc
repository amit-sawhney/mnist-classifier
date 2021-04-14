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

  std::ifstream training_image_stream(
      "../data/datasets/trainingimagesandlabels.txt");

  training_image_stream >> model;

  model.Train();

  float accuracy =
      model.GetAccuracy("../data/datasets/testimagesandlabels.txt");

  std::cout << accuracy;

  return 0;
}
