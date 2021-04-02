#include <core/model.h>
#include "iostream"

namespace naivebayes {

Model::Model() = default;

std::string Model::GetBestClass() const { return "CS 126"; }

void Model::Train(const std::vector<TrainingImage *> &training_images) {}

void Model::Predict() {}

void Model::Load() {}

void Model::Save() {}

std::istream& operator>>(std::istream &in, Model& model) {
  std::string current_line;

  while(std::getline(in, current_line)) {
    if (current_line.length() == 1) {
      std::cout << current_line << std::endl;
    }
  }

  return in;
}

} // namespace naivebayes