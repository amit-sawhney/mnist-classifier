#pragma once

#include <string>
#include <vector>

#include "image.h"
#include "trainer.h"

namespace naivebayes {

/**
 * Represents the Naive Bayes Model to predict numbers for
 */
class Model {
public:
  /**
   * Default constructor
   */
  Model();

  /**
   * Destroys the training map and trainer for a Model
   */
  ~Model();

  /**
   * Copy constructor
   *
   * @param source the incoming Model to copy over
   */
  Model(const Model &source);

  /**
   * Move constructor
   *
   * @param source the incoming Model to steal the data from
   */
  Model(Model &&source) noexcept;

  /**
   * Copy assignment operator
   *
   * @param source the incoming Model to copy over
   * @return the current instance of the Model
   */
  Model &operator=(const Model &source);

  /**
   * Move assignment operator
   *
   * @param source the incoming Model to steal the data from
   * @return the current instance of the Model
   */
  Model &operator=(Model &&source) noexcept;

  /**
   * Trains the current Model with the passed in data through the >> operator
   * override
   *
   * @throws std::exception if the model's training data has not been
   * instantiated
   */
  void Train();

  /**
   * Predicts the classification for an ascii image
   *
   * @param ascii_image the string ascii image representation
   * @return the classification of the image
   */
  char Predict(const std::vector<std::string> &ascii_image);

  /**
   * Predicts the classification for an ascii image
   *
   * @param pixel_grid the pixel based representation of an image
   * @return the classification of the image
   */
  char Predict(const std::vector<std::vector<Pixel>> &pixel_grid);

  /**
   * Deserializes a file back into a Model object
   *
   * @param model_file_path
   */
  void Load(const std::string &model_file_path);

  /**
   * Overrides istream for Model to allow model to be instantiated through the
   * >> operator
   *
   * @param input the input istream that the data will be coming through
   * @param model the model class to populate the input data with
   * @return the input stream
   */
  friend std::istream &operator>>(std::istream &input, Model &model);

  /**
   * Overrides ostream for Trainer to write a custom serialization to
   * an output stream
   *
   * @param output the output stream to write to
   * @param trainer the Trainer to output to the stream
   * @return the output stream
   */
  friend std::ostream &operator<<(std::ostream &output, const Model &trainer);

  /**
   * Adds a training image to the model
   *
   * @param ascii_image the ascii representation of the image
   * @param label the label the image corresponds to
   */
  void AddImage(const std::vector<std::string> &ascii_image, char label);

  /**
   * Determines the Accuracy for a model for a passed in testing dataset
   * filepath
   *
   * @param testing_file_path the dataset of testing images
   * @return the accuracy of the model
   */
  float GetAccuracy(const std::string &testing_file_path);

  /**
   * Calculates the Likelihood value for a singular image corresponding to a
   * specified label
   *
   * @param label the classification to determine the likelihood to
   * @param image the image to calculate the likelihood for
   * @return the value of the calculated likelihood
   */
  float CalculateLikelihood(char label, const Image &image) const;

  Trainer *GetTrainer() const;

  std::map<char, std::vector<Image *>> GetTrainingImageMap() const;

private:
  /**
   * Updates the labels of the training map by instantiating the map with the
   * passed key and an empty vector if the passed label is not part of the map
   *
   * @param label the label to update the map with
   */
  void UpdateTrainingImageMap(char label);

  /**
   * Deletes and clears the data from the current Model object
   */
  void ClearModel();

  std::vector<char> GetLabels() const;

  std::map<char, std::vector<Image *>> label_image_map_;
  Trainer *model_trainer_;
  size_t total_num_images_;
};

} // namespace naivebayes
