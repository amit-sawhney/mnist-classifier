#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() = default;

PredictionMatrix::PredictionMatrix(size_t image_size, size_t num_shades,
                                   size_t num_labels) {

  probabilities_ = StructureMatrix(image_size, num_shades, num_labels);
}

void PredictionMatrix::CalculateProbabilities(
    const std::map<size_t, std::vector<TrainingImage *>> &image_map) {

  std::map<size_t, size_t> test;

  // Traverse each row of an image
  for (size_t i = 0; i < probabilities_.size(); ++i) {

    // Traverse each column of an image
    for (size_t j = 0; j < probabilities_[i].size(); ++j) {

      // Traverse each shade of an pixel
      for (size_t pixel = 0; pixel < probabilities_[i][j].size(); ++pixel) {

        // Traverse each type of image
        for (size_t label = 0; label < probabilities_[i][j][pixel].size();
             ++label) {

          const std::vector<TrainingImage *> &label_images =
              image_map.at(label);
          Pixel current_pixel = ParseSizeTToPixel(pixel);

          size_t num_labels = image_map.size();
          size_t num_images = CalculateNumImagesOfLabelWithPixel(
              i, j, current_pixel, label_images);
          size_t total_images = label_images.size();

          float probability =
              float(kLaplaceSmoothingFactor + num_images) /
              float(num_labels * kLaplaceSmoothingFactor + total_images);

          probabilities_[i][j][pixel][label] = probability;
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