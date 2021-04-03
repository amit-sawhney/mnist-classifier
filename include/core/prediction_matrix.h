#include <map>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class PredictionMatrix {

public:
  PredictionMatrix();

  PredictionMatrix(const std::vector<TrainingImage *> &training_images,
                   size_t image_size, size_t num_shades, size_t num_labels);

  std::vector<std::vector<std::vector<std::vector<float>>>>
  GetPredictionMatrix() const;

private:
  const float kLaplaceSmoothingFactor = 1.0f;

  void BuildMatrix(const std::vector<TrainingImage *> &training_images);

  std::vector<std::vector<std::vector<std::vector<float>>>> probability_matrix_;
  std::map<size_t, size_t> label_to_quantity_map_;
};
} // namespace naivebayes
