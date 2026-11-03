# Milk Adulteration Microwave Sensor 
Video Drive link: https://drive.google.com/file/d/1SuFYMHAHfRGtl58AJcJbWh4Pf5YSGQ50/view?usp=drive_link

Bidirectional Machine Learning & Streamlit Dashboard for microwave resonator (CSRR) sensor characterization.

## Files in this package
- `app.py`: Streamlit interactive web dashboard
- `milk_model.py`: Core bidirectional machine learning engine
- `milk_sensor_dataset.csv`: Sensor measurement dataset (MUST be in the same folder)
- `requirements.txt`: Python package dependencies

## Quick Setup & Run Instructions

### 1. Install Dependencies
Run in your terminal / command prompt:
```bash
pip install -r requirements.txt
```
*(Or if using Windows python launcher: `py -m pip install -r requirements.txt`)*

### 2. Launch Dashboard
```bash
streamlit run app.py
```
*(Or: `py -m streamlit run app.py`)*

### 3. Open in Browser
Open: `http://localhost:8501`

