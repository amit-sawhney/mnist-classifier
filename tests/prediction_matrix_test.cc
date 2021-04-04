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
