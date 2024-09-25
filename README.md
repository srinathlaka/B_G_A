# Bacterial Growth Analysis

This Streamlit application facilitates the analysis of bacterial growth through data upload, background subtraction, growth model fitting, and phase analysis.

## Folder Structure

`bacterial_growth_analysis/`

- `app.py`
- `assets/`
  - `test.xlsx`
- `components/`
  - `file_upload.py`
  - `data_processing.py`
  - `model_fitting.py`
  - `visualization.py`
  - `phase_analysis.py`
- `utils/`
  - `growth_models.py`
  - `metrics.py`

## Description of Components

- `app.py`: Main entry point for the Streamlit application.
- `assets/`
  - `test.xlsx`: Sample data file for testing purposes.
- `components/`:
  - `file_upload.py`: Handles file uploading and reading.
  - `data_processing.py`: Manages data selection and background subtraction.
  - `model_fitting.py`: Functions for fitting models and computing confidence intervals.
  - `visualization.py`: Provides functions for data visualization.
  - `phase_analysis.py`: Facilitates the analysis of growth phases.
- `utils/`:
  - `growth_models.py`: Defines various growth models used in the application.
  - `metrics.py`: Calculates metrics like RSS, R-squared, and AIC.

## Installation and Setup

To set up and run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/srinathlaka/B_G_A.git
    cd bacterial_growth_analysis
    ```

2. Set up a Python virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

After starting the application:
- Upload your data in .xlsx or .csv format.
- Select the layout for your data's well configuration.
- Select wells for analysis via the interface.
- Perform analysis such as model fitting and background subtraction directly through the UI.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
