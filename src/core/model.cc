#include "iostream"
#include <core/model.h>

namespace naivebayes {

Model::Model() = default;

Model::Model(const Model *source) {}

Model::Model(Model &&source) noexcept {}

Model &Model::operator=(const Model &source) { return *this; }

Model &Model::operator=(Model &&source) noexcept { return *this; }

Model::~Model() {
  delete prediction_matrix_;
  for (TrainingImage *image : training_images_) {
    delete image;
  }

  prediction_matrix_ = nullptr;
  training_images_.clear();
}

std::string Model::GetBestClass() const { return "CS 126"; }

void Model::Train() {
  if (training_images_.empty()) {
    throw std::exception("No training images to train the model on");
  }
}

void Model::Predict() {}

void Model::Load(const std::string &model_file_path) {}

void Model::Save(const std::string &save_file_path) {}

std::istream &operator>>(std::istream &input, Model &model) {
  std::string current_line;
  std::vector<std::string> ascii_image;

  size_t current_label = 0;
  while (std::getline(input, current_line)) {
    // Text file is on a line with a label
    if (current_line.length() == 1) {

      // Only create a new image if the data has been collected for it
      if (!ascii_image.empty()) {
        TrainingImage image(ascii_image, current_label);
        TrainingImage *image_ptr = &image;
        ascii_image.clear();
        model.training_images_.push_back(image_ptr);
      }

      current_label = std::stoi(current_line);
      continue;
    }

    ascii_image.push_back(current_line);
  }

  return input;
}

} // namespace naivebayes