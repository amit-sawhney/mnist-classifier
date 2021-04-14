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

  std::cout << model.GetAccuracy("../data/datasets/testimagesandlabels.txt");

  model.PrintConfusionMatrix();

  return 0;
}
