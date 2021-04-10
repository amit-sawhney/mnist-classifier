#include <visualizer/naive_bayes_app.h>

namespace naivebayes {

namespace visualizer {

NaiveBayesApp::NaiveBayesApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin, 2) {
  ci::app::setWindowSize((int)kWindowSize, (int)kWindowSize);

  model_ = Model();
  model_.Load("C:\\Users\\asawh\\Cinder\\my-projects\\naive-bayes-amit-"
              "sawhney\\saved\\saved_model.txt");
}

void NaiveBayesApp::draw() {
  ci::Color8u background_color(255, 246, 148); // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();
  ci::Font text_size("Size Adjustment", 30);

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"), text_size);

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"),
      text_size);
}

void NaiveBayesApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NaiveBayesApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
  case ci::app::KeyEvent::KEY_RETURN: {
    std::vector<std::vector<Pixel>> pixels = sketchpad_.GetPixelGrid();
    current_prediction_ = model_.Predict(pixels) - '0';
    break;
  }
  case ci::app::KeyEvent::KEY_DELETE:
    sketchpad_.Clear();
    break;
  }
}

} // namespace visualizer

} // namespace naivebayes
