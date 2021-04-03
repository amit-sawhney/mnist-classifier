#include <map>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class PredictionMatrix {

public:
  PredictionMatrix();

  PredictionMatrix(size_t image_size, size_t num_shades, size_t num_labels);

  void CalculateProbabilities(
      const std::map<size_t, std::vector<TrainingImage *>> &image_map);

private:
  const float kLaplaceSmoothingFactor = 1.0f;

  void BuildMatrix(const std::vector<TrainingImage *> &training_images);

  size_t CalculateNumImagesOfLabelWithPixel(
      size_t i, size_t j, Pixel pixel,
      const std::vector<TrainingImage *> &images);

  Pixel ParseSizeTToPixel(size_t pixel_num);

  std::vector<std::vector<std::vector<std::vector<float>>>>
  StructureMatrix(size_t image_size, size_t num_shades, size_t num_labels);

  std::vector<std::vector<std::vector<std::vector<float>>>> probabilities_;
};
} // namespace naivebayes
