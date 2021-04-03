#include "core/training_image.h"

namespace naivebayes {
TrainingImage::TrainingImage() = default;

TrainingImage::TrainingImage(const std::vector<std::string> &raw_ascii_image,
                             size_t image_label)
    : image_label_(image_label) {

  // Dimensions of the training image is the length of a row in the ascii image
  image_size_ = raw_ascii_image.at(0).length();

  for (const std::string &image_line : raw_ascii_image) {
    std::vector<Pixel> pixel_row;

    for (char pixel_string : image_line) {

      if (pixel_string == kShadedChar) {
        pixel_row.push_back(Pixel::kShaded);
      } else if (pixel_string == kPartiallyShadedChar) {
        pixel_row.push_back(Pixel::kPartiallyShaded);
      } else {
        pixel_row.push_back(Pixel::kUnshaded);
      }
    }

    pixels_.push_back(pixel_row);

  }
}

Pixel TrainingImage::GetPixelStatusAt(size_t x, size_t y) {
  return pixels_.at(x).at(y);
}

size_t TrainingImage::GetLabel() const { return image_label_; }

size_t TrainingImage::GetSize() const { return image_size_; }
} // namespace naivebayes