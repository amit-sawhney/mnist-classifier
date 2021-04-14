#include <catch2/catch.hpp>

#include <core/model.h>
#include <fstream>
#include <iostream>

using naivebayes::Model;

const std::string kTestTrainingSet =
    "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
    "sawhney\\data\\test_datasets\\test_trainingimagesandlabels.txt";

const std::string kTestSaved =
    "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
    "sawhney\\saved\\testing_saved.txt";

TEST_CASE("Model Default Constructor", "[constructor][trainer]") {
  Model model;

  SECTION("Trainer is null") { REQUIRE(model.GetTrainer() == nullptr); }
}

TEST_CASE("Model Copy Constructor", "[constructor][copy]") {

  SECTION("Empty model to Populated model") {
    Model copy_model;

    std::ifstream saved_model(kTestTrainingSet);

    saved_model >> copy_model;

    Model model(copy_model);

    model.Train();
    copy_model.Train();

    REQUIRE(model.GetTrainer()->GetFeatures() ==
            copy_model.GetTrainer()->GetFeatures());
  }
}

TEST_CASE("Model Move Constructor", "[constructor][move]") {

  SECTION("Empty Model to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(kTestTrainingSet);

    saved_model >> copy_model;

    copy_model.Train();

    Model model = std::move(copy_model);

    model.Train();

    REQUIRE(model.GetTrainer()->GetFeatures()[0][0][0].at('0') ==
            Approx(0.16666667f));
    REQUIRE(copy_model.GetTrainer() == nullptr);
  }
}

TEST_CASE("Model copy assignment operator", "[constructor][copy][operator]") {

  SECTION("Empty to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(kTestTrainingSet);

    saved_model >> copy_model;

    copy_model.Train();

    Model model;
    model = copy_model;

    model.Train();

    REQUIRE(model.GetTrainer()->GetFeatures() ==
            copy_model.GetTrainer()->GetFeatures());
  }

  SECTION("Populated to Empty Model") {
    Model copy_model;

    std::ifstream saved_model(kTestTrainingSet);

    Model model;
    saved_model >> model;

    model = copy_model;

    REQUIRE(model.GetTrainer() == copy_model.GetTrainer());
  }

  SECTION("Populated to Populated Model") {
    Model copy_model;

    std::ifstream saved_model1(kTestTrainingSet);

    Model model;

    saved_model1 >> copy_model;

    std::ifstream saved_model2(kTestTrainingSet);

    saved_model2 >> model;

    model = copy_model;

    model.Train();
    copy_model.Train();

    REQUIRE(model.GetTrainer()->GetFeatures() ==
            copy_model.GetTrainer()->GetFeatures());
  }
}

TEST_CASE("Model move assignment operator", "[constructor][move][operator]") {
  SECTION("Empty to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(kTestTrainingSet);

    saved_model >> copy_model;

    copy_model.Train();

    Model model;
    model = std::move(copy_model);
    model.Train();

    REQUIRE(model.GetTrainer() != nullptr);
    REQUIRE(model.GetTrainer()->GetFeatures()[0][0][0].at('0') ==
            Approx(0.16666666667f));
  }
}

TEST_CASE("Model istream operator", "[operator]") {

  SECTION("Smaller image data set is read correctly") {
    std::ifstream saved_model1(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\sample_ascii_image.txt");

    Model model;
    saved_model1 >> model;

    std::map<char, std::vector<naivebayes::Image *>> map =
        model.GetTrainingImageMap();

    std::vector<naivebayes::Image *> images = map.at('0');

    std::vector<std::vector<naivebayes::Pixel>> expected_pixels = {
        {naivebayes::Pixel::kShaded, naivebayes::Pixel::kShaded,
         naivebayes::Pixel::kShaded},
        {naivebayes::Pixel::kPartiallyShaded,
         naivebayes::Pixel::kPartiallyShaded,
         naivebayes::Pixel::kPartiallyShaded},
        {naivebayes::Pixel::kUnshaded, naivebayes::Pixel::kUnshaded,
         naivebayes::Pixel::kUnshaded}};

    REQUIRE(images.size() == 1);
    REQUIRE(images[0]->GetLabel() == '0');
    REQUIRE(images[0]->GetSize() == 3);
    REQUIRE(images[0]->GetPixels() == expected_pixels);
  }
}

TEST_CASE("Model training", "[train][trainer]") {

  SECTION("Model does not train with no training data") {
    Model model;

    REQUIRE_THROWS(model.Train());
  }

  SECTION("Feature Probabilities are calculated properly") {
    std::ifstream training_data_test(kTestTrainingSet);

    Model model;
    training_data_test >> model;

    model.Train();

    naivebayes::Trainer trainer(3, 3, {'0', '1'});
    naivebayes::FeatureVector expected_values = trainer.GetFeatures();

    expected_values[0][0][0]['0'] = 0.166667f;
    expected_values[0][0][0]['1'] = 0.333333f;
    expected_values[0][0][1]['0'] = 0.333333f;
    expected_values[0][0][1]['1'] = 0.416667f;
    expected_values[0][0][2]['0'] = 0.5f;
    expected_values[0][0][2]['1'] = 0.25f;
    expected_values[0][1][0]['0'] = 0.166667f;
    expected_values[0][1][0]['1'] = 0.0833333f;
    expected_values[0][1][1]['0'] = 0.5f;
    expected_values[0][1][1]['1'] = 0.333333f;
    expected_values[0][1][2]['0'] = 0.333333f;
    expected_values[0][1][2]['1'] = 0.583333f;
    expected_values[0][2][0]['0'] = 0.166667f;
    expected_values[0][2][0]['1'] = 0.833333f;
    expected_values[0][2][1]['0'] = 0.333333f;
    expected_values[0][2][1]['1'] = 0.0833333f;
    expected_values[0][2][2]['0'] = 0.5f;
    expected_values[0][2][2]['1'] = 0.0833333f;
    expected_values[1][0][0]['0'] = 0.166667f;
    expected_values[1][0][0]['1'] = 0.833333f;
    expected_values[1][0][1]['0'] = 0.5f;
    expected_values[1][0][1]['1'] = 0.0833333f;
    expected_values[1][0][2]['0'] = 0.333333f;
    expected_values[1][0][2]['1'] = 0.0833333f;
    expected_values[1][1][0]['0'] = 0.666667f;
    expected_values[1][1][0]['1'] = 0.0833333f;
    expected_values[1][1][1]['0'] = 0.166667f;
    expected_values[1][1][1]['1'] = 0.583333f;
    expected_values[1][1][2]['0'] = 0.166667f;
    expected_values[1][1][2]['1'] = 0.333333f;
    expected_values[1][2][0]['0'] = 0.166667f;
    expected_values[1][2][0]['1'] = 0.833333f;
    expected_values[1][2][1]['0'] = 0.5f;
    expected_values[1][2][1]['1'] = 0.0833333f;
    expected_values[1][2][2]['0'] = 0.333333f;
    expected_values[1][2][2]['1'] = 0.0833333f;
    expected_values[2][0][0]['0'] = 0.166667f;
    expected_values[2][0][0]['1'] = 0.583333f;
    expected_values[2][0][1]['0'] = 0.333333f;
    expected_values[2][0][1]['1'] = 0.166667f;
    expected_values[2][0][2]['0'] = 0.5f;
    expected_values[2][0][2]['1'] = 0.25f;
    expected_values[2][1][0]['0'] = 0.166667f;
    expected_values[2][1][0]['1'] = 0.0833333f;
    expected_values[2][1][1]['0'] = 0.5f;
    expected_values[2][1][1]['1'] = 0.333333f;
    expected_values[2][1][2]['0'] = 0.333333f;
    expected_values[2][1][2]['1'] = 0.583333f;
    expected_values[2][2][0]['0'] = 0.166667f;
    expected_values[2][2][0]['1'] = 0.583333f;
    expected_values[2][2][1]['0'] = 0.333333f;
    expected_values[2][2][1]['1'] = 0.166667f;
    expected_values[2][2][2]['0'] = 0.5f;
    expected_values[2][2][2]['1'] = 0.25f;

    naivebayes::FeatureVector model_trainer = model.GetTrainer()->GetFeatures();

    for (size_t row = 0; row < model_trainer.size(); ++row) {

      for (size_t col = 0; col < model_trainer[row].size(); ++col) {

        for (size_t pixel = 0; pixel < model_trainer[row][col].size();
             ++pixel) {

          for (auto &label_itr : model_trainer[row][col][pixel]) {

            REQUIRE(expected_values[row][col][pixel][label_itr.first] ==
                    Approx(model_trainer[row][col][pixel][label_itr.first]));
          }
        }
      }
    }
  }

  SECTION("Prior Probabilities are calculated properly") {
    std::ifstream training_data_test(kTestTrainingSet);

    Model model;
    training_data_test >> model;

    model.Train();

    naivebayes::Trainer trainer(3, 3, {'0', '1'});
    std::map<char, float> expected_values = trainer.GetPriors();

    expected_values['0'] = 0.285714f;
    expected_values['1'] = 0.714286f;

    std::map<char, float> priors = model.GetTrainer()->GetPriors();

    REQUIRE(priors.size() == 2);
    REQUIRE(expected_values.at('0') == Approx(priors.at('0')));
    REQUIRE(expected_values.at('1') == Approx(priors.at('1')));
  }
}

TEST_CASE("Saving and loading model", "[save][trainer][ostream]") {

  SECTION("Model is saved in correct format") {

    std::ifstream input_file(kTestTrainingSet);

    Model model;
    input_file >> model;
    model.Train();

    model.Save(kTestSaved, "testing_saved.txt");

    Model saved_model;

    saved_model.Load(kTestSaved);

    REQUIRE(model.GetTrainer()->GetFeatures().size() ==
            saved_model.GetTrainer()->GetFeatures().size());
    REQUIRE(model.GetTrainer()->GetFeatures()[0][0][0].at('0') ==
            model.GetTrainer()->GetFeatures()[0][0][0].at('0'));
  }
}