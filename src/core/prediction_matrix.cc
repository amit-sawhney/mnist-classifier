#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() {
  probabilities_ = StructureMatrix(0, 0, 0);
};

PredictionMatrix::PredictionMatrix(size_t image_size, size_t num_shades,
                                   size_t num_labels) {

  probabilities_ = StructureMatrix(image_size, num_shades, num_labels);
}

std::vector<std::vector<std::vector<std::vector<float>>>>
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
  size_t num_labels = std::stoi(current_line);

  matrix.probabilities_ =
      matrix.StructureMatrix(image_size, num_shades, num_labels);

  // Traverse each row of an image
  for (size_t row = 0; row < matrix.probabilities_.size(); ++row) {

    // Traverse each column of an image
    for (size_t col = 0; col < matrix.probabilities_[row].size(); ++col) {

      // Traverse each shade of an pixel
      for (size_t pixel = 0; pixel < matrix.probabilities_[row][col].size();
           ++pixel) {

        // Traverse each type of image
        for (size_t label = 0;
             label < matrix.probabilities_[row][col][pixel].size(); ++label) {

          std::getline(input, current_line);

          matrix.probabilities_[row][col][pixel][label] =
              std::stof(current_line);
        }
      }
    }
  }

  return input;
}

std::ostream &operator<<(std::ostream &output, const PredictionMatrix &matrix) {

  std::vector<std::vector<std::vector<std::vector<float>>>> probabilities =
      matrix.probabilities_;

  for (size_t row = 0; row < probabilities.size(); ++row) {

    for (size_t col = 0; col < probabilities[row].size(); ++col) {

      for (size_t pixel = 0; pixel < probabilities[row][col].size(); ++pixel) {

        for (size_t label = 0; label < probabilities[row][col][pixel].size();
             ++label) {

          output << probabilities[row][col][pixel][label] << std::endl;
        }
      }
    }
  }

  return output;
}

void PredictionMatrix::CalculateProbabilities(
    const std::map<size_t, std::vector<TrainingImage *>> &image_map) {

  for (size_t row = 0; row < probabilities_.size(); ++row) {

    for (size_t col = 0; col < probabilities_[row].size(); ++col) {

      for (size_t pixel = 0; pixel < probabilities_[row][col].size(); ++pixel) {

        for (size_t label = 0; label < probabilities_[row][col][pixel].size();
             ++label) {

          const std::vector<TrainingImage *> &label_images =
              image_map.at(label);
          Pixel current_pixel = ParseSizeTToPixel(pixel);

          size_t num_labels = image_map.size();
          size_t num_images = CalculateNumImagesOfLabelWithPixel(
              row, col, current_pixel, label_images);
          size_t total_images = label_images.size();

          float probability =
              float(kLaplaceSmoothingFactor + num_images) /
              float(num_labels * kLaplaceSmoothingFactor + total_images);

          probabilities_[row][col][pixel][label] = probability;
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

void PredictionMatrix::ClearValues() { probabilities_.clear(); }

std::vector<std::vector<std::vector<std::vector<float>>>>
PredictionMatrix::StructureMatrix(size_t image_size, size_t num_shades,
                                  size_t num_labels) {

  std::vector<float> labels(num_labels, 0.0f);
  std::vector<std::vector<float>> shades(num_shades, labels);
  std::vector<std::vector<std::vector<float>>> y_matrix(image_size, shades);
  std::vector<std::vector<std::vector<std::vector<float>>>> matrix(image_size,
                                                                   y_matrix);

  return matrix;
}
} // namespace naivebayes