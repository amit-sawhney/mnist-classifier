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

TEST_CASE("Training Image explicit Constructor",
          "[constructor][image_size][image_label][pixels]") {

  SECTION("Invalid pixel dimensions") {}

  SECTION("Incorrect matching of image size and pixel vector") {}

  SECTION("Correct field initializations") {}
}

TEST_CASE("Training Image Implicit Constructor", "[ascii_image][image_label]") {

  SECTION("Invalid ascii_image dimensions") {}

  SECTION("Correct field initializations") {}
}

TEST_CASE("Training Image Copy Constructor", "[constructor][copy]") {

  SECTION("Empty to Populated Training Image") {}

  SECTION("Populated to Empty Training Image") {}
}

TEST_CASE("Training Image Move Constructor", "[constructor][move]") {

  SECTION("Empty to Populated Training Image") {}

  SECTION("Populated to Empty Training Image") {}
}

TEST_CASE("Training Image Copy Assignment operator",
          "[constructor][operator][copy]") {

  SECTION("Empty to Populated Training Image") {}

  SECTION("Populated to Empty Training Image") {}

  SECTION("Populated to Populated Training Image") {}
}

TEST_CASE("Training Image Move Assignment operator") {

  SECTION("Empty to Populated Training Image") {}

  SECTION("Populated to Empty Training Image") {}

  SECTION("Populated to Populated Training Image") {}
}

TEST_CASE("Get Pixel Status", "[pixels]") {

  SECTION("Invalid row index") {}

  SECTION("Invalid column index") {}

  SECTION("Correct pixel values") {}
}
