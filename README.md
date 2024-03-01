# BatteryInsight: State of Health (SoH) Analysis and Predictive Modeling



## Introduction

BatteryInsight is a comprehensive project that focuses on the analysis and predictive modeling of State of Health (SoH) for Li-Ion batteries. The project utilizes machine learning and deep learning techniques, including Support Vector Regression (SVR) and Long Short-Term Memory (LSTM) neural networks, to forecast battery life and voltage based on available data.

## Motivation

The motivation behind BatteryInsight lies in the critical importance of battery life and remaining capacity. Understanding battery health is crucial for optimizing usage, extending battery life, and ensuring reliable performance in various applications, from electronic devices to electric vehicles.

## Dataset

### Origin

The dataset used in this project is sourced from NASA PCoE, specifically experiments on Li-Ion batteries. The data includes information on charging and discharging cycles performed at different temperatures, with impedance recorded as a damage criterion.

### Structure

The dataset is initially in MATLAB format and is pre-processed to extract relevant information. The structured data is stored in CSV format, including separate files for charging and discharging cycles. These files are then merged, cleaned, and enhanced with State of Health (SoH) for use in data analysis and machine learning models.

## Exploratory Data Analysis (EDA)

Battery behavior is visualized through various plots, such as Capacity per Cycle and State of Health per Cycle, using both Matplotlib and Plotly Express. These visualizations aid in understanding the patterns and trends in battery performance.

### Capacity per Cycle

![Capacity per Cycle](pics/capacity_per_cycle.png)

This plot illustrates the variation in battery capacity across cycles, providing insights into the degradation pattern.

### State of Health per Cycle

![State of Health per Cycle](path.png)

The State of Health per Cycle plot showcases how battery health evolves over charging and discharging cycles.

## Predictive Modeling

BatteryInsight employs both Support Vector Regression (SVR) and Long Short-Term Memory (LSTM) neural networks for predictive modeling. The models are trained on the pre-processed data to forecast battery life and voltage.

### SVR and LSTM Models

The SVR model is designed to predict battery life, while the LSTM neural network forecasts voltage. Both models are trained and saved for later use in the Streamlit application.

## Streamlit Application

The interactive Streamlit application allows users to select specific batteries for detailed analysis. Users can explore plots of Capacity and State of Health per Cycle, visualize model predictions, and gain valuable insights into battery behavior.

## Conclusion

BatteryInsight provides a comprehensive solution for the analysis and predictive modeling of battery health. The combination of data visualization, machine learning models, and an interactive interface through Streamlit makes it a valuable tool for understanding and optimizing Li-Ion battery performance.

## Languages and Tools

- **Programming Language:** Python
- **IDE:** Visual Studio Code
- **Machine Learning Libraries:** TensorFlow, scikit-learn
- **Visualization Libraries:** Matplotlib, Plotly, Seaborn
- **Web Application Framework:** Streamlit
- **Notebook Environment:** Jupyter Notebooks

