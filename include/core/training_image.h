#include <string>
#include <vector>

#include "enums/pixel.h"

namespace naivebayes {

class TrainingImage {

public:
  TrainingImage();

  TrainingImage(const std::vector<std::string> &raw_ascii_image,
                size_t image_label);

  Pixel GetPixelStatusAt(size_t x, size_t y);

  size_t GetSize() const;

  size_t GetLabel() const;

private:
  const char kShadedChar = '#';
  const char kPartiallyShadedChar = '+';
  size_t image_size_;
  size_t image_label_;
  std::vector<std::vector<Pixel>> pixels_;
};
} // namespace naivebayes
