#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

#include "core/trainer.h"

using naivebayes::Trainer;

TEST_CASE("Prediction Matrix default constructor",
          "[constructor][prediction_matrix]") {
  Trainer matrix;

  SECTION("Probability matrix has no values") {
    REQUIRE(matrix.GetPredictionMatrix().empty());
  }
}

TEST_CASE("Predication Matrix standard constructor", "[constructor]") {

  SECTION("Empty Predication Matrix") {
    Trainer matrix(0, 0, {});

    REQUIRE(matrix.GetPredictionMatrix().empty());
  }

  SECTION("Positive integer dimensions") {
    Trainer matrix(1, 1, {'1'});

    REQUIRE(matrix.GetPredictionMatrix().size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0].size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0][0].size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0][0][0].size() == 1);
  }

  SECTION("Matrix values are initialized to 0") {
    Trainer matrix(1, 1, {});

    REQUIRE(matrix.GetPredictionMatrix()[0][0][0][0] == 0.0f);
  }
}

TEST_CASE("Istream operator overload", "[istream]") {

  SECTION("Invalid information provided") {
    std::ifstream input_file("../data/empty_training_set.txt");

    Trainer matrix;

    REQUIRE_THROWS(input_file >> matrix);
  }

  SECTION("Invalid file structure") {
    std::ifstream input_file("../data/bad_training_set.txt");

    Trainer matrix;

    REQUIRE_THROWS(input_file >> matrix);
  }

  SECTION("Data is properly read in") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer matrix;

    input_file >> matrix;

    auto test_matrix = matrix.GetPredictionMatrix();

    size_t image_size = 3;
    size_t num_pixels = 2;

    for (size_t width = 0; width < image_size; ++width) {
      for (size_t height = 0; height < image_size; ++height) {
        for (size_t pixel = 0; pixel < num_pixels; ++pixel) {
          for (auto itr : test_matrix[width][height][pixel]) {
            REQUIRE(itr.second == Approx(0.05555));
          }
        }
      }
    }
  }

  SECTION("Empty probability data") {
    std::ifstream input_file("../data/empty_training_set.txt");

    Trainer matrix;

    REQUIRE_THROWS(input_file >> matrix);
  }
}

TEST_CASE("Ostream operator overload", "[ostream]") {

  SECTION("Empty Probabilities") {

    Trainer matrix(2, 2, {'0', '1'});

    std::stringstream os_stream;
    os_stream << matrix;
    std::string output_string = os_stream.str();

    std::string expected_string =
        "0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n";

    REQUIRE(output_string == expected_string);
  }

  SECTION("Probabilities properly outputted to ostream") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer matrix;
    input_file >> matrix;

    std::stringstream os_stream;
    os_stream << matrix;
    std::string output_string = os_stream.str();

    std::string expected = "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n"
                           "0.05555\n";

    REQUIRE(output_string.size() == expected.size());
    REQUIRE(output_string == expected);
  }
}

TEST_CASE("Clear values", "[prediction_matrix]") {

  SECTION("Populated prediction matrix is properly cleared") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer matrix;
    input_file >> matrix;

    std::stringstream os_stream;
    os_stream << matrix;
    std::string output_string = os_stream.str();

    matrix.ClearValues();

    REQUIRE(matrix.GetPredictionMatrix().empty());
  }

  SECTION("Empty prediction matrix is cleared with no error") {
    Trainer matrix(2, 1, {'0'});

    REQUIRE_NOTHROW(matrix.ClearValues());
    REQUIRE(matrix.GetPredictionMatrix().empty());
  }
}
