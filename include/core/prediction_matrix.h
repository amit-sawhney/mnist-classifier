#include <map>
#include <vector>

namespace naivebayes {

class PredictionMatrix {

public:
  PredictionMatrix();

  std::vector<std::vector<std::map<size_t, float>>> GetPredictionMatrix() const;

private:
  std::vector<std::vector<std::vector<std::vector<float>>>> probability_matrix_;
  std::map<size_t, size_t> label_to_quantity_map_;
};
} // namespace naivebayes
