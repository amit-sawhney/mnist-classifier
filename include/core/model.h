#include <string>
#include <vector>

#include "prediction_matrix.h"
#include "training_image.h"

namespace naivebayes {

class Model {
public:
  Model();

  std::string GetBestClass() const;

  void Train();

  void Predict();

  void Save(const std::string &save_file_path);

  void Load(const std::string &model_file_path);

  friend std::istream &operator>>(std::istream &input, Model &model);

private:
  const float kLaplaceSmoothingFactor = 1.0f;
  const size_t kNumClasses = 10;
  std::vector<TrainingImage *> training_images_;
  PredictionMatrix *prediction_matrix_;
};

} // namespace naivebayes
