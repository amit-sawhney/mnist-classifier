#include "core/trainer.h"
#include <iostream>

namespace naivebayes {
Trainer::Trainer() { features_ = BuildStructure(0, 0, std::vector<char>{}); }

Trainer::Trainer(size_t image_size, size_t num_shades,
                 const std::vector<char> &labels) {

  features_ = BuildStructure(image_size, num_shades, labels);
}

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::GetFeatures() const {
  return features_;
}

std::map<char, float> Trainer::GetPriors() const { return priors_; }

std::istream &operator>>(std::istream &input, Trainer &trainer) {

  std::string current_line;
  size_t size = trainer.GetNextSizeT(input);
  size_t shades = trainer.GetNextSizeT(input);
  size_t num_labels = trainer.GetNextSizeT(input);

  std::vector<char> labels = trainer.GetFileLabels(input, num_labels);
  trainer.labels_ = labels;

  std::getline(input, current_line);

  std::map<char, float> priors = trainer.GetFilePriors(input, labels);
  trainer.priors_ = priors;
  trainer.features_ = trainer.BuildStructure(size, shades, labels);
  size_t features = 0;

  for (size_t row = 0; row < trainer.features_.size(); ++row) {
    for (size_t col = 0; col < trainer.features_[row].size(); ++col) {
      for (size_t pixel = 0; pixel < trainer.features_[row][col].size();
           ++pixel) {
        for (auto &label_itr : trainer.features_[row][col][pixel]) {
          std::getline(input, current_line);
          label_itr.second = std::stof(current_line);
          ++features;
        }
      }
    }
  }

  trainer.ValidateInputSize(size, shades, labels.size(), features);
  return input;
}

void Trainer::ValidateInputSize(size_t size, size_t num_shades,
                                size_t num_labels, size_t num_features) const {

  size_t expected_features = size * size * num_shades * num_labels;
  bool isValid = expected_features == num_features;

  if (!isValid) {
    throw std::invalid_argument("Bad file provided");
  }
}

std::map<char, float> Trainer::GetFilePriors(std::istream &input,
                                             std::vector<char> labels) {
  std::string current_line;

  std::map<char, float> priors;

  for (char label : labels) {
    std::getline(input, current_line);
    priors[label] = std::stof(current_line);
  }

  return priors;
}

std::vector<char> Trainer::GetFileLabels(std::istream &input,
                                         size_t num_labels) {
  std::vector<char> labels(num_labels, 0);
  std::string current_line;

  for (char &label : labels) {
    std::getline(input, current_line);
    label = current_line[0];
  }

  return labels;
}

size_t Trainer::GetNextSizeT(std::istream &input) {
  std::string line;
  std::getline(input, line);
  return std::stoi(line);
}

std::ostream &operator<<(std::ostream &output, const Trainer &trainer) {

  std::vector<std::vector<std::vector<std::map<char, float>>>> features =
      trainer.features_;

  std::map<char, float> priors = trainer.priors_;

  for (auto &prior_itr : priors) {
    output << prior_itr.second << std::endl;
  }

  for (size_t row = 0; row < features.size(); ++row) {
    for (size_t col = 0; col < features[row].size(); ++col) {
      for (size_t pixel = 0; pixel < features[row][col].size(); ++pixel) {
        for (const auto &label_itr : features[row][col][pixel]) {
          output << label_itr.second << std::endl;
        }
      }
    }
  }

  return output;
}

void Trainer::CalculateFeatures(
    const std::map<char, std::vector<Image *>> &image_map) {

  for (size_t row = 0; row < features_.size(); ++row) {
    for (size_t col = 0; col < features_[row].size(); ++col) {
      for (size_t pixel = 0; pixel < features_[row][col].size(); ++pixel) {
        for (auto &label_itr : features_[row][col][pixel]) {

          const std::vector<Image *> &images = image_map.at(label_itr.first);
          Pixel current_pixel = kPixelMap.at(pixel);

          size_t num_images =
              CountImagesWithPixel(row, col, current_pixel, images);

          float feature =
              float(kLaplace + num_images) /
              float(size_t(Pixel::kNumShades) * kLaplace + images.size());

          label_itr.second = feature;
        }
      }
    }
  }
}

void Trainer::CalculatePriors(
    const std::map<char, std::vector<Image *>> &image_map,
    size_t total_num_images) {

  for (auto &image_itr : image_map) {
    priors_[image_itr.first] =
        float(kLaplace + image_itr.second.size()) /
        float(image_map.size() * kLaplace + total_num_images);
  }
}

size_t Trainer::CountImagesWithPixel(size_t row, size_t col, Pixel pixel,
                                     const std::vector<Image *> &images) {
  size_t num_images = 0;

  for (Image *image : images) {
    if (image->GetPixelStatusByLocation(row, col) == pixel) {
      ++num_images;
    }
  }

  return num_images;
}

void Trainer::ClearValues() { features_.clear(); }

FeatureVector Trainer::BuildStructure(size_t image_size, size_t num_shades,
                                      const std::vector<char> &all_labels) {

  std::map<char, float> labels;

  for (char label : all_labels) {
    labels[label] = 0.0f;
  }

  std::vector<std::map<char, float>> shades(num_shades, labels);
  std::vector<std::vector<std::map<char, float>>> trainer_y(image_size, shades);
  FeatureVector trainer(image_size, trainer_y);

  return trainer;
}

std::vector<char> Trainer::GetLabels() const { return labels_; }
} // namespace naivebayes