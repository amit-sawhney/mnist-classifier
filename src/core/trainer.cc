#include "core/trainer.h"
#include <iostream>

namespace naivebayes {
Trainer::Trainer() { features_ = BuildStructure(0, 0, std::vector<char>{}); };

Trainer::Trainer(size_t image_size, size_t num_shades,
                 const std::vector<char> &labels) {

  features_ = BuildStructure(image_size, num_shades, labels);
}

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::GetTrainer() const {
  return features_;
}

std::istream &operator>>(std::istream &input, Trainer &trainer) {

  std::string current_line;

  // Collect the model information at the top of the file
  std::getline(input, current_line);
  size_t size = std::stoi(current_line);

  std::getline(input, current_line);
  size_t shades = std::stoi(current_line);

  std::getline(input, current_line);
  size_t num_labels = std::stoi(current_line);
  std::vector<char> labels(num_labels, 0);

  for (char &label : labels) {
    std::getline(input, current_line);
    label = current_line[0];
  }

  std::getline(input, current_line);

  std::map<char, float> priors;
  for (char label : labels) {
    std::getline(input, current_line);
    priors[label] = std::stof(current_line);
  }

  trainer.priors_ = priors;
  trainer.features_ = trainer.BuildStructure(size, shades, labels);

  size_t expected_features = size * size * shades * labels.size();
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

  if (expected_features != features || features == 0) {
    throw std::invalid_argument("Bad file provided");
  }

  return input;
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

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::BuildStructure(size_t image_size, size_t num_shades,
                        const std::vector<char> &all_labels) {

  std::map<char, float> labels;

  for (char label : all_labels) {
    labels[label] = 0.0f;
  }
  std::vector<std::map<char, float>> shades(num_shades, labels);
  std::vector<std::vector<std::map<char, float>>> trainer_y(image_size, shades);
  std::vector<std::vector<std::vector<std::map<char, float>>>> trainer(
      image_size, trainer_y);

  return trainer;
}
} // namespace naivebayes