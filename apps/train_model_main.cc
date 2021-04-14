#include "string"
#include <iostream>

#include <algorithm>
#include <core/model.h>
#include <fstream>

int main() {

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
