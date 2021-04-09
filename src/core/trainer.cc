#include "core/trainer.h"
#include <iostream>

namespace naivebayes {
Trainer::Trainer() {
  probabilities_ = StructureMatrix(0, 0, std::vector<char>{});
};

Trainer::Trainer(size_t image_size, size_t num_shades,
                                   const std::vector<char> &labels) {

  probabilities_ = StructureMatrix(image_size, num_shades, labels);
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
  std::string split_delimiter = " ";
  std::vector<char> labels = matrix.Split(current_line, split_delimiter);

  std::map<char, float> prior_probabilities;
  for (size_t prior_count = 0; prior_count < labels.size(); ++prior_count) {
    std::getline(input, current_line);
    prior_probabilities[labels[prior_count]] = std::stof(current_line);
  }

  matrix.prior_probabilities_ = prior_probabilities;

  size_t expected_num_probabilities =
      image_size * image_size * num_shades * labels.size();
  size_t count_probabilities = 0;

  matrix.probabilities_ =
      matrix.StructureMatrix(image_size, num_shades, labels);

  // Traverse each row of an image
  for (size_t row = 0; row < matrix.probabilities_.size(); ++row) {
    // Traverse each column of an image
    for (size_t col = 0; col < matrix.probabilities_[row].size(); ++col) {
      // Traverse each shade of an pixel
      for (size_t pixel = 0; pixel < matrix.probabilities_[row][col].size();
           ++pixel) {
        // Traverse each type of image
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
          Pixel current_pixel = ParseSizeTToPixel(pixel);

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

size_t Trainer::CalculateNumImageLabelsByPixel(
    size_t i, size_t j, Pixel pixel,
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

Pixel Trainer::ParseSizeTToPixel(size_t pixel_num) {
  switch (pixel_num) {
  case 0:
    return Pixel::kUnshaded;
  case 1:
    return Pixel::kPartiallyShaded;
  case 2:
    return Pixel::kShaded;
  default:
    throw std::invalid_argument("Invalid Pixel Num");
  }
}

std::vector<std::vector<std::vector<std::map<char, float>>>>
Trainer::StructureMatrix(size_t image_size, size_t num_shades,
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

std::vector<char> Trainer::Split(std::string string,
                                          const std::string &delimiter) {

  size_t delimiter_idx;
  std::string section;

  std::vector<char> split_string;

  while ((delimiter_idx = string.find(delimiter)) != std::string::npos) {
    section = string.substr(0, delimiter_idx);
    if (section.size() > 1) {
      throw std::invalid_argument("Unable to convert to character");
    }

    char character = section[0];
    split_string.push_back(character);
    string.erase(0, delimiter_idx + delimiter.size());
  }

  return split_string;
}
} // namespace naivebayes