#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() = default;

PredictionMatrix::PredictionMatrix(
    const std::map<size_t, std::vector<TrainingImage *>> &image_map,
    size_t image_size, size_t num_shades, size_t num_labels) {

  std::vector<float> labels(num_labels, 0.0f);

  std::vector<std::vector<float>> shades(num_shades, labels);

  std::vector<std::vector<std::vector<float>>> y_matrix(image_size, shades);

  std::vector<std::vector<std::vector<std::vector<float>>>> matrix(image_size,
                                                                   y_matrix);
  probability_matrix_ = matrix;
}

std::vector<std::vector<std::vector<std::vector<float>>>>
PredictionMatrix::GetPredictionMatrix() const {
  std::vector<std::vector<std::vector<std::vector<float>>>> vector;

  return vector;
}
void PredictionMatrix::BuildMatrix(
    const std::vector<TrainingImage *> &training_images) {}
} // namespace naivebayes