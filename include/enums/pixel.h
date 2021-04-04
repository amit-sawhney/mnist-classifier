
namespace naivebayes {

/**
 * Represents a singular the status of a singular Pixel in an image
 */
enum class Pixel {
  kUnshaded,
  kPartiallyShaded,
  kShaded,
  kNumShades = 3,
};
} // namespace naivebayes