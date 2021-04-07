#include <string>
#include <vector>

#include "prediction_matrix.h"
#include "training_image.h"

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
   * Destroys the training map and prediction matrix for a Model
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

  // TODO: This will be implemented in week 2
  void Predict();

  /**
   * Serializes the Model into a file at the specified location
   *
   * @param save_file_path the location to save the
   * @param file_name the name of the saved file
   */
  void Save(const std::string &save_file_path, const std::string &file_name);

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

  PredictionMatrix *GetPredictionMatrix() const;

  std::map<char, std::vector<TrainingImage *>> GetTrainingImageMap() const;

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

  std::map<char, std::vector<TrainingImage *>> label_training_image_map_;
  PredictionMatrix *prediction_matrix_;
  size_t total_num_images_;
};

} // namespace naivebayes
