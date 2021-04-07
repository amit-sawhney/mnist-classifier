#include <map>
#include <vector>

#include "training_image.h"

namespace naivebayes {
/**
 * Represents a Matrix of values for the Model to predict on
 */
class PredictionMatrix {

public:
  /**
   * Default Constructor
   */
  PredictionMatrix();

  /**
   * Initializes the values dimensions and structure of the probability matrix
   *
   * @param image_size the size of the images going into the matrix
   * @param num_shades the number of shades
   * @param labels all of the labels that the model was trained on
   */
  PredictionMatrix(size_t image_size, size_t num_shades,
                   const std::vector<char> &labels);

  /**
   * Overrides ostream for Prediction Matrix to write a custom serialization to
   * an output stream
   *
   * @param output the output stream to write to
   * @param matrix the Prediction Matrix to output to the stream
   * @return the output stream
   */
  friend std::ostream &operator<<(std::ostream &output,
                                  const PredictionMatrix &matrix);

  /**
   * Overrides the istream operator for the Prediction Matrix to load and
   * deserialize a serialized output
   *
   * @param input the input stream to read in
   * @param matrix the Prediction Matrix to populate
   * @return the input stream
   */
  friend std::istream &operator>>(std::istream &input,
                                  PredictionMatrix &matrix);

  /**
   * Sets all of the probabilities within the probability matrix
   *
   * @param image_map the set of training images mapped to their label
   */
  void CalculateProbabilities(
      const std::map<char, std::vector<TrainingImage *>> &image_map);

  /**
   * Clears all of the values in the probability matrix
   */
  void ClearValues();

  std::vector<std::vector<std::vector<std::map<char, float>>>>
  GetPredictionMatrix() const;

private:
  const float kLaplaceSmoothingFactor = 1.0f;

  /**
   * Calculates the number of Images of a specific label with a specified pixel
   * status
   *
   * @param i the row position in the image
   * @param j the column position in the image
   * @param pixel the pixel status of the location in the image
   * @param images the set of all of the images pertaining to that label
   * @return the number of images matching the specified qualifications
   */
  static size_t
  CalculateNumImageLabelsByPixel(size_t i, size_t j, Pixel pixel,
                                 const std::vector<TrainingImage *> &images);

  /**
   * Parses a numerical representation of a Pixel into an enumeration
   *
   * @param pixel_num the numerical representation
   * @return the pixel enumeration corresponding to this numerical value
   */
  Pixel ParseSizeTToPixel(size_t pixel_num);

  /**
   * Initializes the probability matrix structure as specified by the parameters
   *
   * @param image_size the size of the images going into the matrix
   * @param num_shades the number of shades
   * @param num_labels the number of distinct classifications of numbers
   * @return a multidimensional vector representing the probability matrix
   */
  std::vector<std::vector<std::vector<std::map<char, float>>>>
  StructureMatrix(size_t image_size, size_t num_shades,
                  const std::vector<char> &all_labels);

  std::vector<char> Split(std::string string, const std::string &delimiter);

  std::vector<std::vector<std::vector<std::map<char, float>>>> probabilities_;
};
} // namespace naivebayes
