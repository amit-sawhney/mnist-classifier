#include <catch2/catch.hpp>

#include "core/training_image.h"

using naivebayes::TrainingImage;

TEST_CASE("Training Image Default Constructor",
          "[constructor][image_size][image_label]") {
  TrainingImage image;

  SECTION("Image label is properly instantiated") {
    REQUIRE(image.GetLabel() == INT_MAX);
  }

  SECTION("Image size is 0") { REQUIRE(image.GetSize() == 0); }
}
