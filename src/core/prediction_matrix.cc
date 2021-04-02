#include "core/prediction_matrix.h"

namespace naivebayes {
PredictionMatrix::PredictionMatrix() = default;

std::vector<std::vector<std::map<size_t, float>>>
PredictionMatrix::GetPredictionMatrix() const {}
} // namespace naivebayes