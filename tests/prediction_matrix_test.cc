#include <catch2/catch.hpp>

#include "core/prediction_matrix.h"

using naivebayes::PredictionMatrix;

TEST_CASE("Prediction Matrix default constructor",
          "[constructor][prediction_matrix]") {
  PredictionMatrix matrix;

  SECTION("Probability matrix has no values") {
    REQUIRE(matrix.GetPredictionMatrix().empty());
  }
}

TEST_CASE("Predication Matrix standard constructor", "[constructor]") {

  SECTION("Empty Predication Matrix") {
    PredictionMatrix matrix(0, 0, {});

    REQUIRE(matrix.GetPredictionMatrix().empty());
  }

  SECTION("Positive integer dimensions") {
    PredictionMatrix matrix(1, 1, {'1'});

    REQUIRE(matrix.GetPredictionMatrix().size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0].size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0][0].size() == 1);
    REQUIRE(matrix.GetPredictionMatrix()[0][0][0].size() == 1);
  }

  SECTION("Matrix values are initialized to 0") {
    PredictionMatrix matrix(1, 1, {});
    
    REQUIRE(matrix.GetPredictionMatrix()[0][0][0][0] == 0.0f);
  }
}

TEST_CASE("Istream operator overload", "[istream]") {

  SECTION("Invalid information provided") {}

  SECTION("Invalid file format") {}

  SECTION("Data is properly read in") {}

  SECTION("Empty probability data") {}
}

TEST_CASE("Ostream operator overload", "[ostream]") {

  SECTION("Empty Probabilities") {}

  SECTION("Probabilities properly outputted to ostream") {}
}

TEST_CASE("Calculate Probabilities", "[training_image_map]") {

  SECTION("Empty Training Image map") {}

  SECTION("Probabilities are properly calculated") {}
}

TEST_CASE("Clear values", "[prediction_matrix]") {

  SECTION("Empty prediction matrix is cleared with no error") {}

  SECTION("Populated prediction matrix is properly cleared") {}
}
