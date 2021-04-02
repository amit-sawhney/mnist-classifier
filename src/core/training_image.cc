#include "core/training_image.h"

namespace naivebayes {
TrainingImage::TrainingImage() = default;

TrainingImage::TrainingImage(size_t image_size, size_t image_label)
    : image_label_(image_label), image_size_(image_size){};

Pixel *TrainingImage::GetPixelStatusAt(size_t x, size_t y) {}

void TrainingImage::GetLabel() const {}

void TrainingImage::GetSize() const {}
} // namespace naivebayes