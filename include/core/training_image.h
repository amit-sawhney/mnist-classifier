#pragma once
#include <string>
#include <vector>

#include "enums/pixel.h"

namespace naivebayes {

/**
 * Represents one a Training Image for the Model
 */
class TrainingImage {

public:
  /**
   * Default Constructor
   */
  TrainingImage();

  /**
   * Clears all of thee data in a Training Image
   */
  ~TrainingImage();

  /**
   * Copy Constructor
   *
   * @param source the Training Image to copy the data from
   */
  TrainingImage(const TrainingImage &source);

  /**
   * Move constructor
   *
   * @param source the Training Image to move the data from into the current
   * Training Image
   */
  TrainingImage(TrainingImage &&source) noexcept;

  /**
   * Copy assignment operator
   *
   * @param source the Training Image to copy the data from
   * @return the current instance of the Training Image
   */
  TrainingImage &operator=(const TrainingImage &source);

  /**
   * Move assignment operator
   * @param source the Training Image to move the data from into the current
   * Training Image
   * @return the current instance of the Training Image
   */
  TrainingImage &operator=(TrainingImage &&source) noexcept;

  /**
   * Instantiates a Training Image with by directly setting the data member
   * values of a Training Image
   *
   * @param image_size the size of the Training Image
   * @param image_label the label the Training Image represents
   * @param pixels the status of each of the pixels in the Training Image
   */
  TrainingImage(size_t image_size, char image_label,
                const std::vector<std::vector<Pixel>> &pixels);

  /**
   * Instantiates a Training Image by parsing the necessary information from a
   * string based representation of the Training Image
   *
   * @param raw_ascii_image the string representation of the Training Image
   * @param image_label the label the Training Image represents
   */
  TrainingImage(const std::vector<std::string> &raw_ascii_image,
                char image_label);

  /**
   * Gets the Pixel status at a specific row and column number of the Training
   * Image
   *
   * @param row the row number of the pixel
   * @param col the column number of the pixel
   * @return the Pixel enumeration value of at the specified location
   */
  Pixel GetPixelStatusByLocation(size_t row, size_t col);

  size_t GetSize() const;

  char GetLabel() const;

  std::vector<std::vector<Pixel>> GetPixels() const;

private:
  const char kShadedChar = '#';
  const char kPartiallyShadedChar = '+';

  size_t image_size_{};
  char image_label_{};
  std::vector<std::vector<Pixel>> pixels_;
};
} // namespace naivebayes
