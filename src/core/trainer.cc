#include "core/trainer.h"
#include <iostream>

namespace naivebayes {
Trainer::Trainer() {
  probabilities_ = BuildStructure(0, 0, std::vector<char>{});
};

Trainer::Trainer(size_t image_size, size_t num_shades,
                 const std::vector<char> &labels) {

  probabilities_ = BuildStructure(image_size, num_shades, labels);
}

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::GetPredictionMatrix() const {
  return probabilities_;
}

std::istream &operator>>(std::istream &input, Trainer &matrix) {

  std::string current_line;

  // Collect the model information at the top of the file
  std::getline(input, current_line);
  size_t image_size = std::stoi(current_line);

  std::getline(input, current_line);
  size_t num_shades = std::stoi(current_line);

  std::getline(input, current_line);
  size_t num_labels = std::stoi(current_line);
  std::vector<char> labels(num_labels, 0);

  for (char &label : labels) {
    std::getline(input, current_line);
    label = current_line[0];
  }

  std::getline(input, current_line);

  std::map<char, float> prior_probabilities;
  for (char label : labels) {
    std::getline(input, current_line);
    prior_probabilities[label] = std::stof(current_line);
  }

  matrix.prior_probabilities_ = prior_probabilities;
  matrix.probabilities_ = matrix.BuildStructure(image_size, num_shades, labels);

  size_t expected_num_probabilities =
      image_size * image_size * num_shades * labels.size();
  size_t count_probabilities = 0;

  for (size_t row = 0; row < matrix.probabilities_.size(); ++row) {
    for (size_t col = 0; col < matrix.probabilities_[row].size(); ++col) {
      for (size_t pixel = 0; pixel < matrix.probabilities_[row][col].size();
           ++pixel) {
        for (auto &label_itr : matrix.probabilities_[row][col][pixel]) {
          std::getline(input, current_line);
          ++count_probabilities;
          label_itr.second = std::stof(current_line);
        }
      }
    }
  }

  if (expected_num_probabilities != count_probabilities ||
      count_probabilities == 0) {
    throw std::invalid_argument("Bad file provided");
  }

  return input;
}

std::ostream &operator<<(std::ostream &output, const Trainer &matrix) {

  std::vector<std::vector<std::vector<std::map<char, float>>>> probabilities =
      matrix.probabilities_;

  std::map<char, float> prior_probabilities = matrix.prior_probabilities_;

  for (auto &prior_itr : prior_probabilities) {
    output << prior_itr.second << std::endl;
  }

  for (size_t row = 0; row < probabilities.size(); ++row) {
    for (size_t col = 0; col < probabilities[row].size(); ++col) {
      for (size_t pixel = 0; pixel < probabilities[row][col].size(); ++pixel) {
        for (const auto &label_itr : probabilities[row][col][pixel]) {
          output << label_itr.second << std::endl;
        }
      }
    }
  }

  return output;
}

void Trainer::CalculateProbabilities(
    const std::map<char, std::vector<Image *>> &image_map,
    size_t total_num_images) {

  CalculatePriorProbabilities(image_map, total_num_images);

  for (size_t row = 0; row < probabilities_.size(); ++row) {

    for (size_t col = 0; col < probabilities_[row].size(); ++col) {

      for (size_t pixel = 0; pixel < probabilities_[row][col].size(); ++pixel) {
        for (auto &label_itr : probabilities_[row][col][pixel]) {
          const std::vector<Image *> &label_images =
              image_map.at(label_itr.first);
          Pixel current_pixel = kPixelMap.at(pixel);

          size_t num_images = CalculateNumImageLabelsByPixel(
              row, col, current_pixel, label_images);
          size_t total_images = label_images.size();

          float probability =
              float(kLaplaceSmoothingFactor + num_images) /
              float(size_t(Pixel::kNumShades) * kLaplaceSmoothingFactor +
                    total_images);

          label_itr.second = probability;
        }
      }
    }
  }
}

void Trainer::CalculatePriorProbabilities(
    const std::map<char, std::vector<Image *>> &image_map,
    size_t total_num_images) {

  for (auto &image_itr : image_map) {
    prior_probabilities_[image_itr.first] =
        float(kLaplaceSmoothingFactor + image_itr.second.size()) /
        float(image_map.size() * kLaplaceSmoothingFactor + total_num_images);
  }
}

size_t
Trainer::CalculateNumImageLabelsByPixel(size_t i, size_t j, Pixel pixel,
                                        const std::vector<Image *> &images) {

  size_t num_images = 0;

  for (Image *image : images) {
    if (image->GetPixelStatusByLocation(i, j) == pixel) {
      ++num_images;
    }
  }

  return num_images;
}

void Trainer::ClearValues() { probabilities_.clear(); }

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::BuildStructure(size_t image_size, size_t num_shades,
                        const std::vector<char> &all_labels) {

  std::map<char, float> labels;

  for (char label : all_labels) {
    labels[label] = 0.0f;
  }
  std::vector<std::map<char, float>> shades(num_shades, labels);
  std::vector<std::vector<std::map<char, float>>> y_matrix(image_size, shades);
  std::vector<std::vector<std::vector<std::map<char, float>>>> matrix(
      image_size, y_matrix);

  return matrix;
}
} // namespace naivebayes