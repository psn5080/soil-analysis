# Soil Analysis

Author: Pranav S Narayanan  
Copyright (c) [2022] [\[Pranav S Narayanan\]](https://buzzpranav.github.io)  
License: MIT License

## Introduction

All living things depend on soil. Healthy soil is essential for healthy plant growth, human nutrition, water filtration, and much more. Healthy soil also prevents droughts, floods, fires, and even regulates the Earth's climate. Inarguably, the most important use of soil is its use in agriculture. From the first human civilizations to the modern day, agriculture has been the root of the economy, and soil is the root of agriculture. Anyone can throw a few seeds into the ground and get some crops, but to be economically viable and truly helpful for humanity, we must use high-quality soil with proper concentrations of nutrients, along with fertilizers, and proper agriculture techniques to get the maximum crop yield. Having been in agriculture for thousands of years, farmers have definitely mastered the best techniques of it.

## The Problem

With the techniques of agriculture mastered, the only thing standing between humans and the solution to world hunger, climate change, and crop surplus is knowing about how much nutrients, fertilizers, etc. is already in the soil, and how much more is needed for the soil. Fertilizers, artificial yield boosters, and nutrients, when given in an overdosed amount, will be harmful to plants, as well as everything that consumes them. So how do we find this balance between underusing and overdosing fertilizers? The answer lies in soil analysis. Imagine you just take a handful of dirt, send it to a lab, and immediately find out the nutrient composition. With access to this kind of technology, we could easily maximize crop yield. There’s just one problem: No farmer would want to pay a couple of hundred dollars, go through the hassle of packing and sending the soil to a lab, waiting a couple of days for the analysis, only to be thrown a massive report which makes no sense to them.

## Our Solution

We believe that our program and the machine learning algorithm at its core could be the next solution. Take a picture of your soil. Wait 60 seconds, and presto, you get a fully detailed, easy-to-understand report on soil pH values, Organic Matter levels, Organic Carbon, and EC level using state-of-the-art deep learning computer vision algorithms. Combine this information with the ages of experience and knowledge from farmers and we get the most powerful and viable agriculture economy the world has ever seen. This may seem very complex, or even impossible, but it's not. Artificial Intelligence and Deep Learning, at its heart, is just finding patterns, mixed with some statistics and probability. All you need to do is take a picture and leave the rest up to our program.

## Methodology

After taking a picture, the image quality will be enhanced by an algorithm called CLAHE, which basically increases the image contrast without adding noise or other artificial brighteners. The image is then formatted into a square so that the program can easily perform calculations. Using the enhanced square soil image as a base, the program then breaks the image into various color spaces, such as RGB, HSV, LAB, and XYZ. These color spaces help the program accurately reproduce the image in multiple formats and see the image from various perspectives. Using an artificial intelligence model known as Random Forest, the program predicts the pH value and electrical conductivity using the red, green, and blue values. We can find the amount of clay in the soil using the formula: `(-0.0853 * saturation) + 37.1`. Using the clay percentage and hue, we can also find the organic carbon level by plugging in the values to the formula: `0.05262 * hue + 0.11041 * clay + 2.76983`. With the help of a similar formula, we can find the Bulk Density and Organic Matter levels. Using the soil type given by the user and information it has found, the system can classify the best crops to grow. All this, in less than 60 seconds. The program can predict all the values with 92.4% accuracy on loam soil and 86.8% accuracy on other types of soil. What’s more is that the code is completely free and open source, while being licensed, meaning absolutely anyone can see the internal working of the code and suggest improvements. The license prevents large brands from using the code to make it a paid service, ensuring that the program will always be free and transparent.

World Hunger. Climate Change. Poverty. Food Shortage. All fixed, one picture at a time.

## Features

- **Image Enhancement**: Uses CLAHE to enhance image quality.
- **Color Space Conversion**: Converts images into various color spaces (RGB, HSV, LAB, XYZ).
- **Soil Analysis**: Predicts pH value, Organic Matter levels, Organic Carbon, and Electrical Conductivity using machine learning models.
- **Clay Percentage Calculation**: Calculates clay percentage using saturation values.
- **Bulk Density Calculation**: Calculates bulk density using clay and saturation inputs.
- **Organic Carbon and Matter Levels**: Finds organic carbon and matter levels using hue and clay properties.
- **Pretrained Models**: Uses pretrained models to predict soil properties.
- **User-Friendly Reports**: Generates easy-to-understand reports for farmers.

## Usage

1. **Setup**: Ensure you have all the required libraries installed.
2. **Image Capture**: Take a picture of the soil.
3. **Run Analysis**: Run the program to analyze the soil composition.
4. **Receive Report**: Get a detailed report on soil pH values, Organic Matter levels, Organic Carbon, and EC level.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/soil-analysis.git
    ```
2. Navigate to the project directory:
    ```sh
    cd soil-analysis
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct and the process for submitting pull requests.