#include "core/training_image.h"

namespace naivebayes {
TrainingImage::TrainingImage() : image_label_(INT_MAX), image_size_(0) {}

TrainingImage::TrainingImage(const TrainingImage &source) {
  image_size_ = source.image_size_;
  image_label_ = source.image_label_;
  pixels_ = source.pixels_;
}

TrainingImage::TrainingImage(TrainingImage &&source) noexcept {
  *this = std::move(source);
}

TrainingImage &TrainingImage::operator=(const TrainingImage &source) {
  *this = TrainingImage(source);

  return *this;
}

TrainingImage &TrainingImage::operator=(TrainingImage &&source) noexcept {
  *this = source;

  source.image_size_ = 0;
  source.image_label_ = 0;
  source.pixels_.clear();

  return *this;
}

TrainingImage::~TrainingImage() {
  image_size_ = 0;
  image_label_ = 0;
  pixels_.clear();
}

TrainingImage::TrainingImage(size_t image_size, size_t image_label,
                             const std::vector<std::vector<Pixel>> &pixels) {

  image_size_ = image_size;
  image_label_ = image_label;
  pixels_ = pixels;
}

TrainingImage::TrainingImage(const std::vector<std::string> &raw_ascii_image,
                             size_t image_label)
    : image_label_(image_label) {

  // Dimensions of the training image is the length of a row in the ascii image
  image_size_ = raw_ascii_image.at(0).length();

  for (const std::string &image_line : raw_ascii_image) {
    std::vector<Pixel> pixel_row;

    for (char pixel_char : image_line) {

      if (pixel_char == kShadedChar) {
        pixel_row.push_back(Pixel::kShaded);
      } else if (pixel_char == kPartiallyShadedChar) {
        pixel_row.push_back(Pixel::kPartiallyShaded);
      } else {
        pixel_row.push_back(Pixel::kUnshaded);
      }
    }

    pixels_.push_back(pixel_row);
  }
}

Pixel TrainingImage::GetPixelStatusAt(size_t row, size_t col) {
  return pixels_.at(row).at(col);
}

size_t TrainingImage::GetLabel() const { return image_label_; }

size_t TrainingImage::GetSize() const { return image_size_; }

} // namespace naivebayes