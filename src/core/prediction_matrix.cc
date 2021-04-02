#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() = default;

std::vector<std::vector<std::vector<std::vector<float>>>>
PredictionMatrix::GetPredictionMatrix() const {
  std::vector<std::vector<std::vector<std::vector<float>>>> vector;

  return vector;
}
} // namespace naivebayes