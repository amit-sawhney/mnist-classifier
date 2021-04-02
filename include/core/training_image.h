#include <string>
#include <vector>

#include "enums/pixel.h"

namespace naivebayes {

class TrainingImage {

public:

  TrainingImage();

  TrainingImage(size_t image_size, size_t image_label);

  Pixel* GetPixelStatusAt(size_t x, size_t y);

  void GetSize() const;

  void GetLabel() const;
private:
  static size_t num_labels_;
  size_t image_size_;
  size_t image_label_;
  std::vector<std::vector<Pixel*>> pixels_;
};
} // namespace naivebayes
