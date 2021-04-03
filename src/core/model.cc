#include "iostream"
#include <core/model.h>

namespace naivebayes {

Model::Model() = default;

Model::Model(const Model &source) { *this = source; }

Model::Model(Model &&source) noexcept { *this = std::move(source); }

Model &Model::operator=(const Model &source) {
  ClearModel();

  label_training_image_map_ = std::map<size_t, std::vector<TrainingImage *>>();
  prediction_matrix_ = nullptr;

  for (const auto &itr : source.label_training_image_map_) {
    std::vector<TrainingImage *> images = itr.second;

    for (TrainingImage *image : images) {

      TrainingImage *new_image = new TrainingImage(*image);
      label_training_image_map_[new_image->GetLabel()].push_back(new_image);
    }
  }

  prediction_matrix_ = source.prediction_matrix_;

  return *this;
}

Model &Model::operator=(Model &&source) noexcept {
  ClearModel();

  label_training_image_map_ = std::move(source.label_training_image_map_);
  prediction_matrix_ = source.prediction_matrix_;

  source.label_training_image_map_.clear();
  source.prediction_matrix_->ClearValues();

  return *this;
}

Model::~Model() { ClearModel(); }

void Model::Train() {
  if (label_training_image_map_.empty()) {
    throw std::exception("No training images to train the model on");
  }

  // Access the first training image's size
  size_t image_size =
      label_training_image_map_.begin()->second.at(0)->GetSize();

  prediction_matrix_ = new PredictionMatrix(
      image_size, size_t(Pixel::kNumShades), label_training_image_map_.size());

  prediction_matrix_->CalculateProbabilities(label_training_image_map_);
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
        TrainingImage *image = new TrainingImage(ascii_image, current_label);
        ascii_image.clear();
        model.label_training_image_map_[current_label].push_back(image);
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

void Model::ClearModel() {
  delete prediction_matrix_;

  for (const auto &itr : label_training_image_map_) {
    std::vector<TrainingImage *> images = itr.second;

    for (TrainingImage *image : images) {
      delete image;
    }

    images.clear();
  }

  label_training_image_map_.clear();
}

} // namespace naivebayes