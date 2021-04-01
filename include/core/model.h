#include <string>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class Model {
 public:
  std::string GetBestClass() const;
  void Train(const std::vector<TrainingImage>& training_images);
  void Save();
  void Load();
  void Predict();

  friend std::istream& operator>>(const std::istream &os, Model& rhs);

private:
  const float kLaplaceSmoothingFactor = 1.0f;
  const size_t kNumClasses = 10;
  std::vector<std::vector<float>> prediction_matrix_;

};

}  // namespace naivebayes
