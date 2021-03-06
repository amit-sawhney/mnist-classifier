#pragma once
#include <string>
#include <vector>

#include "enums/pixel.h"

namespace naivebayes {

/**
 * Represents one a Training Image for the Model
 */
class Image {

public:
  /**
   * Default Constructor
   */
  Image();

  /**
   * Clears all of thee data in a Training Image
   */
  ~Image();

  /**
   * Copy Constructor
   *
   * @param source the Training Image to copy the data from
   */
  Image(const Image &source);

  /**
   * Move constructor
   *
   * @param source the Training Image to move the data from into the current
   * Training Image
   */
  Image(Image &&source) noexcept;

  /**
   * Copy assignment operator
   *
   * @param source the Training Image to copy the data from
   * @return the current instance of the Training Image
   */
  Image &operator=(const Image &source);

  /**
   * Move assignment operator
   * @param source the Training Image to move the data from into the current
   * Training Image
   * @return the current instance of the Training Image
   */
  Image &operator=(Image &&source) noexcept;

  /**
   * Instantiates a Training Image with by directly setting the data member
   * values of a Training Image
   *
   * @param image_size the size of the Training Image
   * @param image_label the label the Training Image represents
   * @param pixels the status of each of the pixels in the Training Image
   */
  Image(size_t image_size, char image_label,
        const std::vector<std::vector<Pixel>> &pixels);

  /**
   * Instantiates a Training Image by parsing the necessary information from a
   * string based representation of the Training Image
   *
   * @param raw_ascii_image the string representation of the Training Image
   * @param image_label the label the Training Image represents
   */
  Image(const std::vector<std::string> &raw_ascii_image, char image_label);

  /**
   * Overloaded operator to load in a single image through a file
   *
   * @param input the input stream to get the image from
   * @param image the image class to load the file into
   * @return the input stream
   */
  friend std::istream &operator>>(std::istream &input, Image &image);

  /**
   * Gets the Pixel status at a specific row and column number of the Training
   * Image
   *
   * @param row the row number of the pixel
   * @param col the column number of the pixel
   * @return the Pixel enumeration value of at the specified location
   */
  Pixel GetPixelStatusByLocation(size_t row, size_t col) const;

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
