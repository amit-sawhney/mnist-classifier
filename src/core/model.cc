#include <core/model.h>

namespace naivebayes {

Model::Model() = default;

std::string Model::GetBestClass() const { return "CS 126"; }

void Model::Train(const std::vector<TrainingImage *> &training_images) {}

void Model::Predict() {}

void Model::Load() {}

void Model::Save() {}

Model &Model::operator>>(const Model &rhs) {
  return *this;
}

} // namespace naivebayes