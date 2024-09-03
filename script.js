// Custom script for Sentinel Hub to highlight water bodies in blue
// and display the rest of the image in black and white.

function setup() {
    return {
      input: ["B02", "B03", "B04", "B08", "B11"],
      output: { bands: 3 }
    };
  }
  
  function evaluatePixel(sample) {
    // Calculate NDWI to isolate water bodies
    let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  
    // If NDWI indicates water, highlight it in blue
    if (ndwi > 0.3) {
      return [0, 0.5, 1]; // Bright blue
    }
  
    // For non-water areas, convert to grayscale
    let grayscale = (sample.B04 + sample.B03 + sample.B02) / 3;
  
    return [grayscale, grayscale, grayscale];
}