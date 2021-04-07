
namespace naivebayes {

/**
 * Represents the status of a singular Pixel in an image
 */
enum class Pixel : size_t {
  kUnshaded,
  kPartiallyShaded,
  kShaded,
  kNumShades,
};
} // namespace naivebayes