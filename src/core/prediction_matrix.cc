#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() {
  probabilities_ = StructureMatrix(0, 0, std::vector<char>{});
};

PredictionMatrix::PredictionMatrix(size_t image_size, size_t num_shades,
                                   std::vector<char> labels) {

  probabilities_ = StructureMatrix(image_size, num_shades, labels);
}

std::vector<std::vector<std::vector<std::map<char, float>>>>
PredictionMatrix::GetPredictionMatrix() const {
  return probabilities_;
}

std::istream &operator>>(std::istream &input, PredictionMatrix &matrix) {

  std::string current_line;

  // Collect the model information at the top of the file
  std::getline(input, current_line);
  size_t image_size = std::stoi(current_line);

  std::getline(input, current_line);
  size_t num_shades = std::stoi(current_line);

  std::getline(input, current_line);
  std::string split_delimiter = " ";
  std::vector<char> labels = matrix.Split(current_line, split_delimiter);

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

          label_itr.second = std::stof(current_line);
        }
      }
    }
  }

  return input;
}

std::ostream &operator<<(std::ostream &output, const PredictionMatrix &matrix) {

  std::vector<std::vector<std::vector<std::map<char, float>>>> probabilities =
      matrix.probabilities_;

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

void PredictionMatrix::CalculateProbabilities(
    const std::map<char, std::vector<TrainingImage *>> &image_map) {

  for (size_t row = 0; row < probabilities_.size(); ++row) {

    for (size_t col = 0; col < probabilities_[row].size(); ++col) {

      for (size_t pixel = 0; pixel < probabilities_[row][col].size(); ++pixel) {

        for (auto &label_itr : probabilities_[row][col][pixel]) {
          const std::vector<TrainingImage *> &label_images =
              image_map.at(label_itr.first);
          Pixel current_pixel = ParseSizeTToPixel(pixel);

          size_t num_labels = image_map.size();
          size_t num_images = CalculateNumImagesOfLabelWithPixel(
              row, col, current_pixel, label_images);
          size_t total_images = label_images.size();

          float probability =
              float(kLaplaceSmoothingFactor + num_images) /
              float(num_labels * kLaplaceSmoothingFactor + total_images);

          label_itr.second = probability;
        }
      }
    }
  }
}

size_t PredictionMatrix::CalculateNumImagesOfLabelWithPixel(
    size_t i, size_t j, Pixel pixel,
    const std::vector<TrainingImage *> &images) {

  size_t num_images = 0;

  for (TrainingImage *image : images) {
    if (image->GetPixelStatusAt(i, j) == pixel) {
      ++num_images;
    }
  }

  return num_images;
}

void PredictionMatrix::ClearValues() { probabilities_.clear(); }

Pixel PredictionMatrix::ParseSizeTToPixel(size_t pixel_num) {
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
PredictionMatrix::StructureMatrix(size_t image_size, size_t num_shades,
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

std::vector<char> PredictionMatrix::Split(std::string string,
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