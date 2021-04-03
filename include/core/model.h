#include <string>
#include <vector>

#include "prediction_matrix.h"
#include "training_image.h"

namespace naivebayes {

class Model {
public:
  Model();

  ~Model();

  Model(const Model *source);

  Model(Model &&source) noexcept;

  Model &operator=(const Model &source);

  Model &operator=(Model &&source) noexcept;

  std::string GetBestClass() const;

  void Train();

  void Predict();

  void Save(const std::string &save_file_path);

  void Load(const std::string &model_file_path);

  friend std::istream &operator>>(std::istream &input, Model &model);

private:
  const size_t kNumClasses = 10;

  void UpdateTrainingLabelMap(size_t label);

  std::vector<TrainingImage *> training_images_;
  std::map<size_t, size_t> training_label_count_map_;
  PredictionMatrix *prediction_matrix_;
};

} // namespace naivebayes
