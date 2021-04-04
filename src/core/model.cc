#include "iostream"
#include <core/model.h>
#include <fstream>

namespace naivebayes {

Model::Model() {
  prediction_matrix_ = nullptr;
  label_training_image_map_ = std::map<size_t, std::vector<TrainingImage *>>();
}

Model::Model(const Model &source) { *this = source; }

Model::Model(Model &&source) noexcept { *this = std::move(source); }

Model &Model::operator=(const Model &source) {
  ClearModel();

  label_training_image_map_ = std::map<size_t, std::vector<TrainingImage *>>();
  prediction_matrix_ = nullptr;

  for (const auto &itr : source.label_training_image_map_) {
    std::vector<TrainingImage *> images = itr.second;

    for (TrainingImage *image : images) {

      // Utilize Training Image copy constructor
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

  std::cout << "Training Model................" << std::endl;

  // Access the first training image's size
  size_t image_size =
      label_training_image_map_.begin()->second.at(0)->GetSize();

  prediction_matrix_ = new PredictionMatrix(
      image_size, size_t(Pixel::kNumShades), label_training_image_map_.size());

  prediction_matrix_->CalculateProbabilities(label_training_image_map_);

  std::cout << "Finished Training................" << std::endl;
}

void Model::Predict() {
  // TODO: Week 2 implementation
}

void Model::Load(const std::string &model_file_path) {
  std::cout << "Loading Model........" << std::endl;

  std::ifstream saved_stream(model_file_path);
  saved_stream >> *prediction_matrix_;

  std::cout << "Finished Loading........." << std::endl;
}

void Model::Save(const std::string &save_file_path,
                 const std::string &file_name) {

  std::cout << "Saving the model........" << std::endl;

  std::string full_path = save_file_path + file_name;

  // Access the first training image's size
  size_t image_size =
      label_training_image_map_.begin()->second.at(0)->GetSize();
  size_t num_shades = size_t(Pixel::kNumShades);
  size_t num_labels = label_training_image_map_.size();

  std::ofstream os(full_path);

  // Save basic model information at top of file
  os << image_size << std::endl;
  os << num_shades << std::endl;
  os << num_labels << std::endl;

  os << *prediction_matrix_;

  os.close();

  std::cout << "Finished Saving........." << std::endl;
}

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

        // File has read an entire image and is at a new label so add image
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
  // Checks if map doesn't contain the label
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