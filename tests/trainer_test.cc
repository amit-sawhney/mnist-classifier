#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

#include "core/trainer.h"

using naivebayes::Trainer;

TEST_CASE("Trainer default constructor", "[constructor][trainer]") {
  Trainer trainer;

  SECTION("Probability trainer has no values") {
    REQUIRE(trainer.GetFeatures().empty());
  }
}

TEST_CASE("Trainer standard constructor", "[constructor]") {

  SECTION("Empty Trainer") {
    Trainer trainer(0, 0, {});

    REQUIRE(trainer.GetFeatures().empty());
  }

  SECTION("Positive integer dimensions") {
    Trainer trainer(1, 1, {'1'});

    REQUIRE(trainer.GetFeatures().size() == 1);
    REQUIRE(trainer.GetFeatures()[0].size() == 1);
    REQUIRE(trainer.GetFeatures()[0][0].size() == 1);
    REQUIRE(trainer.GetFeatures()[0][0][0].size() == 1);
  }

  SECTION("Trainer values are initialized to 0") {
    Trainer trainer(1, 1, {});

    REQUIRE(trainer.GetFeatures()[0][0][0][0] == 0.0f);
  }
}

TEST_CASE("Istream operator overload", "[istream]") {

  SECTION("Invalid information provided") {
    std::ifstream input_file("../data/empty_training_set.txt");

    Trainer trainer;

    REQUIRE_THROWS(input_file >> trainer);
  }

  SECTION("Invalid file structure") {
    std::ifstream input_file("../data/bad_training_set.txt");

    Trainer trainer;

    REQUIRE_THROWS(input_file >> trainer);
  }

  SECTION("Data is properly read in") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer trainer;

    input_file >> trainer;

    auto test_trainer = trainer.GetFeatures();

    size_t image_size = 3;
    size_t num_pixels = 2;

    for (size_t width = 0; width < image_size; ++width) {
      for (size_t height = 0; height < image_size; ++height) {
        for (size_t pixel = 0; pixel < num_pixels; ++pixel) {
          for (auto itr : test_trainer[width][height][pixel]) {
            REQUIRE(itr.second == Approx(0.05555));
          }
        }
      }
    }
  }

  SECTION("Empty probability data") {
    std::ifstream input_file("../data/empty_training_set.txt");

    Trainer trainer;

    REQUIRE_THROWS(input_file >> trainer);
  }
}

TEST_CASE("Ostream operator overload", "[ostream]") {

  SECTION("Empty Probabilities") {

    Trainer trainer(2, 2, {'0', '1'});

    std::stringstream os_stream;
    os_stream << trainer;
    std::string output_string = os_stream.str();

    std::string expected_string =
        "0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n";

    REQUIRE(output_string == expected_string);
  }

  SECTION("Probabilities properly outputted to ostream") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer trainer;
    input_file >> trainer;

    std::stringstream os_stream;
    os_stream << trainer;
    std::string output_string = os_stream.str();

    std::string expected =
        "0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0."
        "05555\n0.05555\n0.05555\n0.05555\n0.05555\n0."
        "05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0."
        "05555\n0.05555\n"
        "0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n"
        "0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0."
        "05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0."
        "05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0"
        ".05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0.05555\n0"
        ".05555\n0.05555\n";

    REQUIRE(output_string.size() == expected.size());
    REQUIRE(output_string == expected);
  }
}

TEST_CASE("Clear values", "[trainer]") {

  SECTION("Populated trainer is properly cleared") {
    std::ifstream input_file("C:\\Users\\asawh\\Cinder\\my-projects\\naive-"
                             "bayes-amit-sawhney\\data\\save_model_test.txt");

    Trainer trainer;
    input_file >> trainer;

    std::stringstream os_stream;
    os_stream << trainer;
    std::string output_string = os_stream.str();

    trainer.ClearValues();

    REQUIRE(trainer.GetFeatures().empty());
  }

  SECTION("Empty trainer is cleared with no error") {
    Trainer trainer(2, 1, {'0'});

    REQUIRE_NOTHROW(trainer.ClearValues());
    REQUIRE(trainer.GetFeatures().empty());
  }
}
