# WebLogAnalysis

WebLogAnalysis is a project designed to detect malicious and legitimate web logs using ensemble machine learning techniques.

## Project Structure

The repository contains the following key components:

- **Data Files**:
  - `Apache_2k_modified.csv`: Dataset used for training and evaluation.

- **Jupyter Notebooks**:
  - `Bagging.ipynb`: Implementation of the Bagging ensemble method.
  - `Boosting.ipynb`: Implementation of the Boosting ensemble method.
  - `Stacking.ipynb`: Implementation of the Stacking ensemble method.

- **Flask Application**:
  - Located in the `flask_app` directory, this contains the web application for deploying the models.

- **Models**:
  - Stored in the `models` directory, this includes pre-trained machine learning models.

## Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Flask

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Rajneeshkarya/WebLogAnalysis.git
   cd WebLogAnalysis
   ```

2. **Install required packages**:

   It's recommended to use a virtual environment. You can install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: Ensure that a `requirements.txt` file is present in the repository with all necessary dependencies listed.

## Usage

### Running the Jupyter Notebooks

To explore the ensemble methods:

1. Navigate to the project directory:

   ```bash
   cd path_to_project_directory
   ```

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open and run the desired notebook (`Bagging.ipynb`, `Boosting.ipynb`, or `Stacking.ipynb`) to see the implementation details and results.

### Running the Flask Application

To deploy the web application:

1. Navigate to the `flask_app` directory:

   ```bash
   cd flask_app
   ```

2. Run the Flask app:

   ```bash
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000/` to interact with the application.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any inquiries or issues, please open an issue in this repository.
