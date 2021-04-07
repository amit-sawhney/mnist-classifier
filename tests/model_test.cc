#include <catch2/catch.hpp>

#include <core/model.h>
#include <fstream>
#include <iostream>

using naivebayes::Model;

TEST_CASE("Model Default Constructor", "[constructor][prediction_matrix]") {
  Model model;

  SECTION("Prediction Matrix is null") {
    REQUIRE(model.GetPredictionMatrix() == nullptr);
  }
}

TEST_CASE("Model Copy Constructor", "[constructor][copy]") {

  SECTION("Empty model to Populated model") {
    Model copy_model;

    std::ifstream saved_model(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    saved_model >> copy_model;

    Model model(copy_model);

    model.Train();
    copy_model.Train();

    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix() ==
            copy_model.GetPredictionMatrix()->GetPredictionMatrix());
  }
}

TEST_CASE("Model Move Constructor", "[constructor][move]") {

  SECTION("Empty Model to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    saved_model >> copy_model;

    copy_model.Train();

    Model model = std::move(copy_model);

    model.Train();

    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix()[0][0][0].at(
                '0') == Approx(0.16666667f));
    REQUIRE(copy_model.GetPredictionMatrix() == nullptr);
  }
}

TEST_CASE("Model copy assignment operator", "[constructor][copy][operator]") {

  SECTION("Empty to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    saved_model >> copy_model;

    copy_model.Train();

    Model model;
    model = copy_model;

    model.Train();

    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix() ==
            copy_model.GetPredictionMatrix()->GetPredictionMatrix());
  }

  SECTION("Populated to Empty Model") {
    Model copy_model;

    std::ifstream saved_model(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    Model model;
    saved_model >> model;

    model = copy_model;

    REQUIRE(model.GetPredictionMatrix() == copy_model.GetPredictionMatrix());
  }

  SECTION("Populated to Populated Model") {
    Model copy_model;

    std::ifstream saved_model1(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    Model model;

    saved_model1 >> copy_model;

    std::ifstream saved_model2(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    saved_model2 >> model;

    model = copy_model;

    model.Train();
    copy_model.Train();

    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix() ==
            copy_model.GetPredictionMatrix()->GetPredictionMatrix());
  }
}

TEST_CASE("Model move assignment operator", "[constructor][move][operator]") {
  SECTION("Empty to Populated Model") {
    Model copy_model;

    std::ifstream saved_model(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    saved_model >> copy_model;

    copy_model.Train();

    Model model;
    model = std::move(copy_model);
    model.Train();

    REQUIRE(model.GetPredictionMatrix() != nullptr);
    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix()[0][0][0].at(
                '0') == Approx(0.16666666667f));
  }
}

TEST_CASE("Model istream operator", "[operator]") {

  SECTION("Smaller image data set is read correctly") {
    std::ifstream saved_model1(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\sample_ascii_image.txt");

    Model model;
    saved_model1 >> model;

    std::map<char, std::vector<naivebayes::TrainingImage *>> map =
        model.GetTrainingImageMap();

    std::vector<naivebayes::TrainingImage *> images = map.at('0');

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

TEST_CASE("Model training", "[train][prediction_matrix]") {

  //  SECTION("Model does not train with no training data") {
  //    Model model;
  //
  //    REQUIRE_THROWS(model.Train());
  //  }

  SECTION("Probabilities are calculated properly") {
    std::ifstream training_data_test(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    Model model;
    training_data_test >> model;

    model.Train();

    std::cout << model.GetPredictionMatrix()->GetPredictionMatrix()[0][0][2].at(
        '0');
  }
}

TEST_CASE("Saving and loading model", "[save][prediction_matrix][ostream]") {

  SECTION("Model is saved in correct format") {

    std::ifstream input_file(
        "C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
        "sawhney\\data\\test_trainingimagesandlabels.txt");

    Model model;
    input_file >> model;
    model.Train();

    model.Save("C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
               "sawhney\\saved\\",
               "testing_saved.txt");

    Model saved_model;

    saved_model.Load("C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
                     "sawhney\\saved\\testing_saved.txt");

    REQUIRE(model.GetPredictionMatrix()->GetPredictionMatrix().size() ==
            saved_model.GetPredictionMatrix()->GetPredictionMatrix().size());
    REQUIRE(
        model.GetPredictionMatrix()->GetPredictionMatrix()[0][0][0].at('0') ==
        model.GetPredictionMatrix()->GetPredictionMatrix()[0][0][0].at('0'));
  }
}