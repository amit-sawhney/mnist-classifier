#include <string>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class Model {
public:
  Model();

  std::string GetBestClass() const;

  void Train(const std::vector<TrainingImage *> &training_images);

  void Predict();

  void Save();

  void Load();

  Model &operator>>(const Model &rhs);

private:
  const float kLaplaceSmoothingFactor = 1.0f;
  const std::string kShadedChar = "#";
  const std::string kPartiallyShadedChar = "+";
  const size_t kNumClasses = 10;
  std::vector<std::vector<float>> prediction_matrix_;
};

} // namespace naivebayes
