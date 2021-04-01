#include <string>
#include <vector>

#include "enums/pixel.h"

namespace naivebayes {

class TrainingImage {

public:
  void GetSize() const;

  void GetLabel() const;
private:
  size_t image_size_;
  size_t image_label_;
  std::vector<std::vector<Pixel*>> pixels_;
};
} // namespace naivebayes
