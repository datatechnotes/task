# Product Prediction Application

## Introduction

Welcome to the Product Prediction Application, a project designed to predict furniture products for a given list of URLs. This application utilizes advanced natural language processing techniques to analyze web page content and determine the presence of furniture products.

### Key Features:

- **Product Prediction:** The core functionality of the application is to predict whether a given target web URL contains furniture products or not.

- **Data Collection:** The application collects training data from a predefined shortlist of furniture stores provided in a CSV file.

- **Model Training:** The collected data is processed and used to train a BERT (Bidirectional Encoder Representations from Transformers) Named Entity Recognition (NER) model. This model is then capable of identifying product-related entities within the text.

### Usage:

1. **Prepare Training Data:**
   - A shortlist of furniture stores is provided in a CSV file (`short furniture stores pages.csv`).
   - The application collects and saves training data in a file (`training_data.csv`).
   - If the train data is already collected, the application skips this step.

2. **Train the Model:**
   - The program trains the BERT NER model using the collected training data.
   - The trained model is saved for future predictions in the (`ner_model`) folder.
   - If the model is already built, the application skips this step too.

3. **Predict Products:**
   - Provide a list of target web URLs in CSV format (`furniture stores pages.csv`).
   - The application predicts whether each URL contains furniture products.

### Getting Started:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Follow the usage instructions to prepare training data, train the model, and predict products.
4. Run the application.
```bash
 python main.py
```

### Results:

The application provides results in the following format:

```code
PRODUCT available in this page! URL: https://www.factorybuys.com.au/products/euro-top-mattress-king
PRODUCT available in this page! URL: https://dunlin.com.au/products/beadlight-cirrus
PRODUCT available in this page! URL: https://themodern.net.au/products/hamar-plant-stand-ash
PRODUCT available in this page! URL: https://interiorsonline.com.au/products/interiors-online-gift-card
```




