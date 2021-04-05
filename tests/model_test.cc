#include <catch2/catch.hpp>

#include <core/model.h>

using naivebayes::Model;

TEST_CASE("Model Default Constructor", "[constructor][prediction_matrix]") {
  Model model;

  SECTION("Prediction Matrix is null") {
    REQUIRE(model.GetPredictionMatrix() == nullptr);
  }
}

TEST_CASE("Model Copy Constructor", "[constructor][copy]") {

  SECTION("Empty model to Populated model") {}

  SECTION("Populated Model to Empty Model") {}
}

TEST_CASE("Model Move Constructor", "[constructor][move]") {

  SECTION("Empty Model to Populated Model") {}

  SECTION("Populated Model to Empty Model") {}
}

TEST_CASE("Model copy assignment operator", "[constructor][copy][operator]") {

  SECTION("Empty to Populated Model") {}

  SECTION("Populated to Empty Model") {}

  SECTION("Populated to Populated Model") {}
}

TEST_CASE("Model move assignment operator", "[constructor][move][operator]") {
  SECTION("Empty to Populated Model") {}

  SECTION("Populated to Empty Model") {}

  SECTION("Populated to Populated Model") {}
}

TEST_CASE("Model istream operator", "[operator]") {

  SECTION("Smaller image data set is read correctly") {}
}

TEST_CASE("Model training", "[train][prediction_matrix]") {

  SECTION("Model does not train with no training data") {}

  SECTION("Model updates prediction matrix with correct values") {}
}

TEST_CASE("Saving model") {

  SECTION("Model creates file in specified directory") {}

  SECTION("Model is saved in correct format") {}
}

TEST_CASE("Loading model") {

  SECTION("Invalid file data format") {}

  SECTION("Model prediction matrix is updated properly") {}
}