#include <catch2/catch.hpp>

#include <core/model.h>

using naivebayes::Model;

TEST_CASE("Model Default Constructor", "[constructor][prediction_matrix]") {
  Model model;

  SECTION("Prediction Matrix is null") {
    REQUIRE(model.GetPredictionMatrix() == nullptr);
  }
}
