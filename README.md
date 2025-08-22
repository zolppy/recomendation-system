# Image Recommendation System using VGG16 and Cosine Similarity

This project demonstrates the creation of a content-based image recommendation system. It leverages a pre-trained VGG16 deep learning model to extract meaningful feature vectors from images. These features are then compared using cosine similarity to identify and recommend visually similar images.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Methodology](https://www.google.com/search?q=%23methodology)
  - [How to Run](https://www.google.com/search?q=%23how-to-run)
  - [Evaluation](https://www.google.com/search?q=%23evaluation)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)

## Project Overview

The core idea is to transform images into a high-dimensional space where their proximity represents visual similarity. A powerful, pre-trained Convolutional Neural Network (CNN) is used for this transformation. Once images are represented as numerical vectors, we can calculate the similarity between them and build a recommendation engine.

The notebook accomplishes the following:

1.  Loads and preprocesses the `tf_flowers` image dataset.
2.  Uses a pre-trained VGG16 model on ImageNet to extract feature vectors from all images.
3.  Implements a recommendation function based on the cosine similarity between these feature vectors.
4.  Evaluates the recommendation system's performance using an average precision metric.

## Dataset

The project uses the **`tf_flowers`** dataset, available through `tensorflow_datasets`.

  - **Total Images:** 3,670
  - **Number of Classes:** 5
  - **Class Names:** `dandelion`, `daisy`, `tulips`, `sunflowers`, `roses`

The dataset is split as follows:

  - **Training Set:** 80%
  - **Validation Set:** 10%
  - **Test Set:** 10%

## Methodology

The workflow is divided into four main stages:

1.  **Data Loading and Preprocessing:**

      - Images are loaded from the `tf_flowers` dataset.
      - Each image is resized to $224 \\times 224$ pixels to match the input dimensions required by the VGG16 model.
      - Pixel values are normalized from the `[0, 255]` range to `[0, 1]`.
      - The datasets are batched for efficient processing.

2.  **Feature Extraction:**

      - A VGG16 model, pre-trained on the ImageNet dataset, is loaded without its final classification layer (`include_top=False`).
      - This base model acts as a powerful feature extractor. Each image is passed through the network, and the output from the last convolutional block is flattened to produce a high-dimensional feature vector.
      - This process is applied to all images in the training, validation, and test sets.

3.  **Similarity Calculation:**

      - **Cosine Similarity** is used to measure the similarity between the feature vectors of two images. It calculates the cosine of the angle between two vectors, providing a score between -1 and 1 (or 0 and 1 for non-negative vectors). A score closer to 1 indicates higher similarity.
      - The formula for cosine similarity between two vectors $A$ and $B$ is:
        $$\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

4.  **Recommendation and Evaluation:**

      - To test the system, random images are selected from the test set to act as queries.
      - For each query image, its feature vector is compared against the feature vectors of all images in the training set.
      - The images from the training set with the highest cosine similarity scores are returned as recommendations.

## How to Run

1.  **Clone the repository or download the `main.ipynb` file.**

2.  **Install the necessary dependencies.** You can install them using pip:

    ```bash
    pip install numpy tensorflow tensorflow-datasets matplotlib scikit-learn
    ```

3.  **Open and run the notebook.**

      - Open the `main.ipynb` file in a Jupyter environment such as JupyterLab, Jupyter Notebook, or Google Colab.
      - Execute the cells in sequential order. The notebook will automatically download the dataset, build the model, extract features, and run the evaluation.

## Evaluation

The performance of the recommendation system is evaluated using **Average Precision**.

For each query image from the test set:

1.  The system retrieves the top 10 most similar images from the training set.
2.  **Precision** is calculated as the proportion of these 10 recommended images that belong to the same class as the query image.
    $$\text{Precision} = \frac{\text{Number of relevant recommendations}}{\text{Total number of recommendations}}$$
3.  The **Average Precision** is the mean of the precision scores calculated for all query images. This final score provides a measure of the system's ability to retrieve visually similar and semantically relevant images.

The notebook's output includes the precision score for each query and the final average precision, along with a brief discussion of the result.

## Dependencies

  - `Python 3.x`
  - `numpy`
  - `tensorflow`
  - `tensorflow_datasets`
  - `matplotlib`
  - `scikit-learn`
