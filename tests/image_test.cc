#include <catch2/catch.hpp>

#include "core/image.h"

using naivebayes::Pixel;
using naivebayes::Image;

TEST_CASE("Training Image Default Constructor",
          "[constructor][image_size][image_label]") {
  Image image;

  SECTION("Image label is properly instantiated") {
    REQUIRE(image.GetLabel() == '\0');
  }

  SECTION("Image size is 0") { REQUIRE(image.GetSize() == 0); }
}

TEST_CASE("Training Image explicit Constructor",
          "[constructor][image_size][image_label][pixels]") {

  SECTION("Invalid pixel dimensions") {
    std::vector<std::vector<Pixel>> pixels{
        {Pixel::kUnshaded, Pixel::kUnshaded}};

    REQUIRE_THROWS(new Image(2, '1', pixels));
  }

  SECTION("Incorrect matching of image size and pixel vector") {
    std::vector<std::vector<Pixel>> pixels{
        {Pixel::kUnshaded, Pixel::kUnshaded},
        {Pixel::kUnshaded, Pixel::kUnshaded}};

    REQUIRE_THROWS(new Image(1, '1', pixels));
  }

  SECTION("Correct field initializations") {
    std::vector<std::vector<Pixel>> pixels{
        {Pixel::kUnshaded, Pixel::kUnshaded},
        {Pixel::kUnshaded, Pixel::kUnshaded}};

    Image image(2, '1', pixels);

    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetPixels() == pixels);
  }
}

TEST_CASE("Training Image Implicit Constructor", "[ascii_image][image_label]") {

  SECTION("Empty ascii_image") {
    std::vector<std::string> ascii_image{};

    REQUIRE_THROWS(new Image(ascii_image, '1'));
  }

  SECTION("Invalid ascii dimensions") {
    std::vector<std::string> ascii_image{"###", "+++"};

    REQUIRE_THROWS(new Image(ascii_image, '1'));
  }

  SECTION("Correct field initializations") {
    std::vector<std::string> ascii_image{"##", "++"};

    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image image(ascii_image, '1');

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);
  }
}

TEST_CASE("Training Image Copy Constructor", "[constructor][copy]") {

  SECTION("Empty to Populated Training Image") {

    std::vector<std::string> ascii_image{"##", "++"};
    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image copy_image(ascii_image, '1');

    Image image(copy_image);

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);
  }
}

TEST_CASE("Training Image Move Constructor", "[constructor][move]") {

  SECTION("Empty to Populated Training Image") {

    std::vector<std::string> ascii_image{"##", "++"};
    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image copy_image(ascii_image, '1');

    Image image = std::move(copy_image);

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);

    REQUIRE(copy_image.GetLabel() == '\0');
    REQUIRE(copy_image.GetSize() == 0);
    REQUIRE(copy_image.GetPixels().empty());
  }
}

TEST_CASE("Training Image Copy Assignment operator",
          "[constructor][operator][copy]") {

  SECTION("Empty to Populated Training Image") {
    Image image;

    std::vector<std::string> ascii_image{"##", "++"};
    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image copy_image(ascii_image, '1');

    image = Image(copy_image);

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);
  }

  SECTION("Populated to Empty Training Image") {

    std::vector<std::string> ascii_image{"##", "++"};

    Image image(ascii_image, '1');
    Image copy_image;

    image = Image(copy_image);

    REQUIRE(image.GetLabel() == '\0');
    REQUIRE(image.GetSize() == 0);
    REQUIRE(image.GetPixels().empty());
  }

  SECTION("Populated to Populated Training Image") {
    std::vector<std::string> ascii_image1{"##", "++"};
    std::vector<std::string> ascii_image2{"###", "+++", "###"};

    std::vector<std::vector<Pixel>> expected_pixels = {
        {Pixel::kShaded, Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded,
         Pixel::kPartiallyShaded},
        {Pixel::kShaded, Pixel::kShaded, Pixel::kShaded}};

    Image image(ascii_image1, '1');
    Image copy_image(ascii_image2, '2');

    image = Image(copy_image);

    REQUIRE(image.GetLabel() == '2');
    REQUIRE(image.GetSize() == 3);
    REQUIRE(image.GetPixels() == expected_pixels);
  }
}

TEST_CASE("Training Image Move Assignment operator") {

  SECTION("Empty to Populated Training Image") {

    std::vector<std::string> ascii_image{"##", "++"};
    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image copy_image(ascii_image, '1');

    Image image;
    image = std::move(copy_image);

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);

    REQUIRE(copy_image.GetLabel() == '\0');
    REQUIRE(copy_image.GetSize() == 0);
    REQUIRE(copy_image.GetPixels().empty());
  }

  SECTION("Populated to Empty Training Image") {
    std::vector<std::string> ascii_image{"##", "++"};

    Image copy_image;

    Image image(ascii_image, '2');
    image = std::move(copy_image);

    REQUIRE(image.GetLabel() == '\0');
    REQUIRE(image.GetSize() == 0);
    REQUIRE(image.GetPixels().empty());

    REQUIRE(copy_image.GetLabel() == '\0');
    REQUIRE(copy_image.GetSize() == 0);
    REQUIRE(copy_image.GetPixels().empty());
  }

  SECTION("Populated to Populated Training Image") {
    std::vector<std::string> ascii_image1{"##", "++"};
    std::vector<std::string> ascii_image2{"###", "+++", "###"};
    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image copy_image(ascii_image1, '1');

    Image image(ascii_image2, '2');
    image = std::move(copy_image);

    REQUIRE(image.GetLabel() == '1');
    REQUIRE(image.GetSize() == 2);
    REQUIRE(image.GetPixels() == expected_pixels);

    REQUIRE(copy_image.GetLabel() == '\0');
    REQUIRE(copy_image.GetSize() == 0);
    REQUIRE(copy_image.GetPixels().empty());
  }
}

TEST_CASE("Get Pixel Status", "[pixels]") {

  SECTION("Invalid row index") {
    std::vector<std::string> ascii_image{"##", "++"};

    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image image(ascii_image, '1');

    REQUIRE_THROWS(image.GetPixelStatusByLocation(2, 0));
  }

  SECTION("Invalid column index") {
    std::vector<std::string> ascii_image{"##", "++"};

    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image image(ascii_image, '1');

    REQUIRE_THROWS(image.GetPixelStatusByLocation(0, 2));
  }

  SECTION("Correct pixel values") {
    std::vector<std::string> ascii_image{"##", "++"};

    std::vector<std::vector<Pixel>> expected_pixels{
        {Pixel::kShaded, Pixel::kShaded},
        {Pixel::kPartiallyShaded, Pixel::kPartiallyShaded}};

    Image image(ascii_image, '1');

    for (size_t row = 0; row < expected_pixels.size(); ++row) {
      for (size_t col = 0; col < expected_pixels[row].size(); ++col) {
        REQUIRE(image.GetPixelStatusByLocation(row, col) == expected_pixels[row][col]);
      }
    }
  }
}
