#pragma once

#include <map>
#include <vector>

#include "image.h"

namespace naivebayes {

typedef std::vector<std::vector<std::vector<std::map<char, float>>>>
    FeatureVector;

/**
 * Represents a Trainer of values for the Model to predict on
 */
class Trainer {

public:
  /**
   * Default Constructor
   */
  Trainer();

  /**
   * Initializes the values dimensions and structure of the trainer
   *
   * @param image_size the size of the images going into the trainer
   * @param num_shades the number of shades
   * @param labels all of the labels that the model was trained on
   */
  Trainer(size_t image_size, size_t num_shades,
          const std::vector<char> &labels);

  /**
   * Overrides ostream for Trainer to write a custom serialization to
   * an output stream
   *
   * @param output the output stream to write to
   * @param trainer the Trainer to output to the stream
   * @return the output stream
   */
  friend std::ostream &operator<<(std::ostream &output, const Trainer &trainer);

  /**
   * Overrides the istream operator for the Trainer to load and
   * deserialize a serialized output
   *
   * @param input the input stream to read in
   * @param trainer the Trainer to populate
   * @return the input stream
   */
  friend std::istream &operator>>(std::istream &input, Trainer &trainer);

  /**
   * Sets all of the probabilities within the trainer
   *
   * @param image_map the set of training images mapped to their label
   */
  void CalculateFeatures(const std::map<char, std::vector<Image *>> &image_map);

  /**
   * Calculates and sets all of the prior probabilities for the trainer
   *
   * @param image_map the map of all of the images to their labels
   * @param total_num_images the total number of images
   */
  void CalculatePriors(const std::map<char, std::vector<Image *>> &image_map,
                       size_t total_num_images);

  /**
   * Clears all of the values in the trainer
   */
  void ClearValues();

  FeatureVector GetFeatures() const;

  std::map<char, float> GetPriors() const;

  std::vector<char> GetLabels() const;

private:
  const float kLaplace = 1.0f;
  const std::map<size_t, Pixel> kPixelMap = {
      {0, Pixel::kUnshaded}, {1, Pixel::kPartiallyShaded}, {2, Pixel::kShaded}};

  /**
   * Calculates the number of Images of a specific label with a specified pixel
   * status
   *
   * @param row the row position in the image
   * @param col the column position in the image
   * @param pixel the pixel status of the location in the image
   * @param images the set of all of the images pertaining to that label
   * @return the number of images matching the specified qualifications
   */
  static size_t CountImagesWithPixel(size_t row, size_t col, Pixel pixel,
                                     const std::vector<Image *> &images);

  /**
   * Initializes the trainer structure as specified by the parameters
   *
   * @param image_size the size of the images going into the trainer
   * @param num_shades the number of shades
   * @param num_labels the number of distinct classifications of numbers
   * @return a multidimensional vector representing the trainer
   */
  FeatureVector BuildStructure(size_t image_size, size_t num_shades,
                               const std::vector<char> &all_labels);

  std::vector<char> GetFileLabels(std::istream &input, size_t num_labels);

  size_t GetNextSizeT(std::istream &input);

  std::map<char, float> GetFilePriors(std::istream &input,
                                      std::vector<char> labels);

  void ValidateInputSize(size_t size, size_t num_shades,
                         size_t num_labels, size_t num_features) const;

  FeatureVector features_;
  std::map<char, float> priors_;
  std::vector<char> labels_;
};
} // namespace naivebayes
