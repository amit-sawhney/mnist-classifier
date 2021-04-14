#include "iostream"
#include <core/model.h>
#include <fstream>

namespace naivebayes {

Model::Model() {
  model_trainer_ = nullptr;
  label_image_map_ = std::map<char, std::vector<Image *>>();
  total_num_images_ = 0;
}

Model::Model(const Model &source) {
  model_trainer_ = nullptr;
  *this = source;
}

Model::Model(Model &&source) noexcept {

  label_image_map_ = std::move(source.label_image_map_);
  model_trainer_ = source.model_trainer_;
  total_num_images_ = source.total_num_images_;

  source.label_image_map_.clear();
  source.model_trainer_ = nullptr;
  source.total_num_images_ = 0;
}

Model &Model::operator=(const Model &source) {

  if (this != &source) {
    ClearModel();

    label_image_map_ = std::map<char, std::vector<Image *>>();
    model_trainer_ = nullptr;

    for (const auto &itr : source.label_image_map_) {
      std::vector<Image *> images = itr.second;

      for (Image *image : images) {

        Image *new_image = new Image(*image);
        label_image_map_[new_image->GetLabel()].push_back(new_image);
      }
    }

    model_trainer_ = source.model_trainer_;
    total_num_images_ = source.total_num_images_;
  }

  return *this;
}

Model &Model::operator=(Model &&source) noexcept {
  ClearModel();

  label_image_map_ = std::move(source.label_image_map_);
  model_trainer_ = source.model_trainer_;
  total_num_images_ = source.total_num_images_;

  source.label_image_map_.clear();

  return *this;
}

Model::~Model() { ClearModel(); }

Trainer *Model::GetTrainer() const { return model_trainer_; }

void Model::Train() {
  if (label_image_map_.empty()) {
    throw std::exception("No training images to train the model on");
  }

  std::cout << "Training Model................" << std::endl;

  // Access the first training image's size
  size_t image_size = label_image_map_.begin()->second.at(0)->GetSize();

  std::vector<char> labels = GetLabels();

  model_trainer_ = new Trainer(image_size, size_t(Pixel::kNumShades), labels);
  model_trainer_->CalculateFeatures(label_image_map_);
  model_trainer_->CalculatePriors(label_image_map_, total_num_images_);

  std::cout << "Finished Training................" << std::endl;
}

char Model::Predict(const std::vector<std::string> &ascii_image) {
  Image predict_image(ascii_image, 0);

  return Predict(predict_image.GetPixels());
}

char Model::Predict(const std::vector<std::vector<Pixel>> &pixel_grid) {
  Image predict_image(pixel_grid.size(), 0, pixel_grid);

  float max_likelihood = INT_MIN;
  char prediction;
  std::vector<char> labels = GetLabels();

  for (char label : labels) {
    float likelihood = CalculateLikelihood(label, predict_image);
    if (likelihood > max_likelihood) {
      max_likelihood = likelihood;
      prediction = label;
    }
  }

  return prediction;
}

float Model::CalculateLikelihood(char label, const Image &image) const {

  auto features = model_trainer_->GetFeatures();
  float sum_probability = 0.0f;
  float prior = model_trainer_->GetPriors().at(label);

  sum_probability += std::log(prior);

  for (size_t row = 0; row < features.size(); ++row) {
    for (size_t col = 0; col < features[row].size(); ++col) {
      size_t pixel = size_t(image.GetPixelStatusByLocation(row, col));
      sum_probability += std::log(features[row][col][pixel].at(label));
    }
  }

  return sum_probability;
}

float Model::GetAccuracy(const std::string &testing_file_path) {
  std::ifstream testing_file(testing_file_path);

  std::string current_line;
  std::vector<std::string> ascii_image;

  std::getline(testing_file, current_line);
  char label = current_line[0];

  size_t total_images = 0;
  size_t correct_predictions = 0;

  while (std::getline(testing_file, current_line)) {
    if (current_line.length() == 1) {

      if (!ascii_image.empty()) {
        char prediction = Predict(ascii_image);

        if (prediction == label) {
          ++correct_predictions;
        }

        ascii_image.clear();
        label = current_line[0];
        ++total_images;
      }
      continue;
    }
    ascii_image.push_back(current_line);
  }

  return float(correct_predictions) / float(total_images);
}

void Model::Load(const std::string &model_file_path) {
  std::cout << "Loading Model........" << std::endl;

  std::ifstream saved_stream(model_file_path);
  model_trainer_ = new Trainer();
  // Overloaded operator to train to load modle
  saved_stream >> *model_trainer_;

  std::cout << "Finished Loading........." << std::endl;
}

std::istream &operator>>(std::istream &input, Model &model) {
  std::string current_line;
  std::vector<std::string> ascii_image;

  std::getline(input, current_line);
  char label = current_line[0];

  while (std::getline(input, current_line)) {
    // Text file is on a line with a label
    if (current_line.length() == 1) {
      model.UpdateTrainingImageMap(label);
      // Only create a new image if the data has been collected for it
      if (!ascii_image.empty()) {
        model.AddImage(ascii_image, label);
        ascii_image.clear();
        label = current_line[0];
      }
      continue;
    }
    ascii_image.push_back(current_line);
  }

  model.AddImage(ascii_image, label);
  model.model_trainer_ = nullptr;

  return input;
}

std::ostream &operator<<(std::ostream &os, const Model &trainer) {
  std::cout << "Saving the model........" << std::endl;

  // Access the first training image's size
  size_t image_size = trainer.label_image_map_.begin()->second.at(0)->GetSize();
  size_t num_shades = size_t(Pixel::kNumShades);
  std::vector<char> labels = trainer.GetLabels();

  // Save basic model information at top of file
  os << image_size << std::endl;
  os << num_shades << std::endl;
  os << trainer.label_image_map_.size() << std::endl;

  for (char label : labels) {
    os << label << std::endl;
  }

  os << std::endl;
  os << *trainer.model_trainer_;
  std::cout << "Finished Saving........." << std::endl;

  return os;
}

void Model::AddImage(const std::vector<std::string> &ascii_image, char label) {
  Image *image = new Image(ascii_image, label);
  label_image_map_[label].push_back(image);
  ++total_num_images_;
}

void Model::UpdateTrainingImageMap(char label) {
  // Initialize a new label in the map with an empty vector
  if (!label_image_map_.count(label)) {
    label_image_map_[label] = std::vector<Image *>{};
    return;
  }
}

void Model::ClearModel() {

  for (const auto &itr : label_image_map_) {
    std::vector<Image *> images = itr.second;

    for (Image *image : images) {
      delete image;
    }

    images.clear();
  }

  delete model_trainer_;
  label_image_map_.clear();
  total_num_images_ = 0;
}

std::vector<char> Model::GetLabels() const {

  if (label_image_map_.empty()) {
    return model_trainer_->GetLabels();
  }

  std::vector<char> labels;

  for (const auto &itr : label_image_map_) {
    labels.push_back(itr.first);
  }

  return labels;
}

std::map<char, std::vector<Image *>> Model::GetTrainingImageMap() const {
  return label_image_map_;
}

} // namespace naivebayes