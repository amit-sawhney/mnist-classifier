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

  for (const auto &itr : label_training_image_map_) {
    std::vector<TrainingImage *> images = itr.second;

    for (TrainingImage *image : images) {
      delete image;
    }

    images.clear();
  }

  label_training_image_map_.clear();

  prediction_matrix_ = nullptr;
  label_training_image_map_.clear();
}

std::string Model::GetBestClass() const { return "CS 126"; }

void Model::Train() {
  if (label_training_image_map_.empty()) {
    throw std::exception("No training images to train the model on");
  }

  // Access the first training image's size
  size_t image_size = label_training_image_map_.begin()->second.size();

  prediction_matrix_ = new PredictionMatrix(
      label_training_image_map_, image_size, size_t(Pixel::kNumShades),
      label_training_image_map_.size());
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

      current_label = std::stoi(current_line);
      model.UpdateTrainingImageMap(current_label);

      // Only create a new image if the data has been collected for it
      if (!ascii_image.empty()) {
        TrainingImage image(ascii_image, current_label);
        TrainingImage *image_ptr = &image;
        ascii_image.clear();
        model.label_training_image_map_[current_label].push_back(image_ptr);
      }

      continue;
    }

    ascii_image.push_back(current_line);
  }

  TrainingImage *image = new TrainingImage(ascii_image, current_label);
  ascii_image.clear();
  model.label_training_image_map_[current_label].push_back(image);

  return input;
}

void Model::UpdateTrainingImageMap(size_t label) {
  if (!label_training_image_map_.count(label)) {
    label_training_image_map_[label] = std::vector<TrainingImage *>{};
    return;
  }
}

} // namespace naivebayes