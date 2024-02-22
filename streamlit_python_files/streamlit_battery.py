import streamlit as st
import pandas as pd
import os 
import warnings
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping


warnings.filterwarnings('ignore')
st.set_page_config(page_title="Forecasting Battery Health: A Predictive Analysis of State of Health (SoH)!", page_icon=":bar_chart:", layout="wide")

# Add decorative elements
st.markdown("<h1 style='text-align: center; color: #4285f4;'>Forecasting Battery Health: A Predictive Analysis of State of Health (SoH)!</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #4285f4;'>", unsafe_allow_html=True)

# plotting capacity per cycle for selected battery
def plot_capacity_per_cycle(df_cleaned, selected_battery_name):
    # Filter the DataFrame based on selected battery name and discharge type
    selected_data = df_cleaned[(df_cleaned['battery_name'] == selected_battery_name) & (df_cleaned['type'] == 'discharge')]
    # max_cycle = selected_data['id_cycle'].max()
    # st.write(f"The maximum id_cycle for {selected_battery_name} (Discharge) is: {max_cycle}")

    if selected_data.empty:
        st.write(f"No data found for {selected_battery_name} with 'discharge' type.")
        return

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(selected_data['id_cycle'], selected_data['Capacity'], marker='o', linestyle='-', color='b', label='Capacity per Cycle')
    plt.title(f'Capacity per Cycle for {selected_battery_name} (Discharge)')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    

def plot_soh_per_cycle(df_cleaned, selected_battery_name):
    # Filter the DataFrame based on selected battery name and discharge type
    selected_data = df_cleaned[(df_cleaned['battery_name'] == selected_battery_name) & (df_cleaned['type'] == 'discharge')]
    max_cycle = selected_data['id_cycle'].max()
    print(f"The maximum id_cycle for {selected_battery_name} (Discharge) is: {max_cycle}")

    if selected_data.empty:
        print(f"No data found for {selected_battery_name} with 'discharge' type.")
        return

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(selected_data['id_cycle'], selected_data['SoH'], marker='o', linestyle='-', color='b', label='SoH per Cycle')
    plt.title(f'SoH per Cycle for {selected_battery_name} (Discharge)')
    plt.xlabel('Cycle')
    plt.ylabel('SoH')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# plotting  voltage_data
def plot_voltage_data(df, selected_battery_name, selected_cycles):
    # Filter the DataFrame based on the selected battery and cycles
    selected_data = df[(df['battery_name'] == selected_battery_name) & (df['id_cycle'] <= selected_cycles)]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(selected_data['id_cycle'], selected_data['Voltage_measured'], marker='^', linestyle='-', color='b')
    plt.title(f'Voltage Measured over {selected_cycles} Charge and Discharge Cycles for Battery {selected_battery_name}')
    plt.xlabel('Cycle')
    plt.ylabel('Voltage Measured')
    plt.grid(True)
    st.pyplot(plt)


# plotting temperature over time for specific battery
def plot_temperature_data(df, selected_battery_name, cycle_num):
    # Filter the DataFrame based on the selected battery and cycles
    selected_data = df[(df['battery_name'] == selected_battery_name) & (df['id_cycle'] == cycle_num)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data['Time'], selected_data['Temperature_measured'], marker='^', linestyle='-', color='b')
    plt.title(f'Temperature Measured over Time for Cycle {cycle_num} - Battery {selected_battery_name}')
    plt.xlabel('CTime')
    plt.ylabel('Temperature_measured')
    plt.grid(True)    
    st.pyplot(plt)
    
    
def plot_voltage_current_charge_over_time(df, battery_name, selected_cycle):
    # Filter data for the selected battery, cycle, and charge type
    selected_data = df[(df['battery_name'] == battery_name) & (df['id_cycle'] == selected_cycle) ]

    # Plot voltage and current over time using Plotly Express
    fig = px.line(selected_data, x='Time', y=['Voltage_measured', 'Current_measured'],
                  title=f'Voltage and Current Over Time - Battery: {battery_name}, Cycle: {selected_cycle} ',
                  labels={'value': 'Measurement', 'variable': 'Parameter', 'Time': 'Time'},
                  line_dash='variable')

    # Update line style for 'Voltage_measured' to blue and bold
    fig.update_traces(line=dict(color='blue', width=2), selector=dict(name='Voltage_measured'))

    # Update line style for 'Current_measured' to red and bold
    fig.update_traces(line=dict(color='red', width=4, dash='solid'), selector=dict(name='Current_measured'))

    # Decrease the height of the graph
    fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', 
                      xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='lightgray', mirror=True,
                                 zeroline=True, zerolinecolor='lightgray', zerolinewidth=2),
                      yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='lightgray', mirror=True, 
                                 zeroline=True, zerolinecolor='lightgray', zerolinewidth=2))
    

    # Show the plot on Streamlit page
    st.plotly_chart(fig)

#######################

def plot_voltage_temperature_over_time(df, battery_name, selected_cycle):
    # Filter data for the selected battery, cycle, and charge type
    selected_data = df[(df['battery_name'] == battery_name) & (df['id_cycle'] == selected_cycle) ]

    # Plot voltage and current over time using Plotly Express
    fig = px.line(selected_data, x='Time', y=['Current_measured', 'Temperature_measured'],
                  title=f'Current and Temperature Over Time - Battery: {battery_name}, Cycle: {selected_cycle} ',
                  labels={'value': 'Measurement', 'variable': 'Parameter', 'Time': 'Time'},
                  line_dash='variable')

    # Update line style for 'Voltage_measured' to blue and bold
    fig.update_traces(line=dict(color='red', width=2), selector=dict(name='Current_measured'))

    # Update line style for 'Current_measured' to red and bold
    fig.update_traces(line=dict(color='green', width=4, dash='solid'), selector=dict(name='Temperature_measured'))

    # Decrease the height of the graph
    fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', 
                      xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='lightgray', mirror=True,
                                 zeroline=True, zerolinecolor='lightgray', zerolinewidth=2),
                      yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='lightgray', mirror=True, 
                                 zeroline=True, zerolinecolor='lightgray', zerolinewidth=2))

    # Show the plot on Streamlit page
    st.plotly_chart(fig)   
    
def plot_correlation_heatmap(df, battery_name):
    # Filter data for the selected battery
    selected_data = df[df['battery_name'] == battery_name]
    # Drop the 'battery_name' and 'type' columns
    selected_data_corr = selected_data.drop(['battery_name',  'ambient_temperature' ], axis=1)
    # Create a correlation matrix
    correlation_matrix = selected_data_corr.corr()
    # Find the two highest correlations
    sorted_correlations = correlation_matrix['Capacity'].sort_values(ascending=False)
    # highest_corr = sorted_correlations.drop_duplicates()[1:3]
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 5))
    # Create a heatmap using Seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap="crest", linewidths=.5, fmt=".2f")
    # Set the title
    plt.title(f'Correlation Heatmap for {battery_name}')    

    # Return the figure and the highest correlations
    return fig, sorted_correlations
    
    
###################################
def train_evaluate_plot(X, y):
    
    plt.figure(figsize=(12, 8))

    # Line plot for actual battery capacity with markers
    plt.plot(X.flatten(), y.flatten(), 'ko-', label='Battery Capacity') 

    ratios = [10, 20, 40, 60, 80]
    palette = sns.color_palette('viridis', n_colors=len(ratios) + 1)
    scores = []

    for i, ratio in enumerate(ratios):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio/100, shuffle=False)
        best_svr = SVR(C=20, epsilon=0.0001, gamma=0.0001, cache_size=200,
                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        best_svr.fit(X_train, y_train)

        y_pred = best_svr.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append((100-ratio, score))

        # Scatter plot for predicted values
        sns.scatterplot(x=X_test.flatten(), y=y_pred.flatten(), label=f'Prediction with train size of {100 - ratio}%', s=50, color=palette[i + 1])

        # Line plot for the predicted values
        sns.lineplot(x=X_test.flatten(), y=y_pred.flatten(), linestyle='dashed', linewidth=2, color=palette[i + 1])

    plt.xlabel('No. of Cycles')
    plt.ylabel('Capacity')
    plt.title('Battery Capacity and Predictions')
    plt.legend()
    plt.show()
    
    st.pyplot(plt)  
    # Return the scores
    return scores
    
 ##################################  
def train_evaluate_plot_param(X, y, ratio=20, C=20, epsilon=0.0001, gamma=0.0001, cache_size=200, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False):
    plt.figure(figsize=(12, 8))

    # Line plot for actual battery capacity with markers
    plt.plot(X.flatten(), y.flatten(), color='mediumblue',marker='o', linewidth=2, markersize=8,linestyle='-', label='Battery Capacity') 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio/100, shuffle=False)
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, cache_size=cache_size,
            kernel=kernel, max_iter=max_iter, shrinking=shrinking, tol=tol, verbose=verbose)
    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_test)
    score = r2_score(y_test, y_pred)

    # Scatter plot for predicted values
    sns.scatterplot(x=X_test.flatten(), y=y_pred.flatten(), label=f'Prediction with train size of {100 - ratio}%', color='red', s=100)

    # Line plot for the predicted values
    sns.lineplot(x=X_test.flatten(), y=y_pred.flatten(), linestyle='dashed', linewidth=2, color='blue')

    plt.xlabel('No. of Cycles')
    plt.ylabel('Capacity')
    plt.title('Battery Capacity and Predictions')
    plt.legend()
    plt.show()
    st.pyplot(plt)  

    # Return the scores
    return score
##################################  
# def train_evaluate_plot_lstm(X, y, max_cycle):
    
#     X_scaler = MinMaxScaler()
#     y_scaler = MinMaxScaler()    
#     X_lstm = X_scaler.fit_transform(X)
#     y_lstm = y_scaler.fit_transform(y)    
#     X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.40, shuffle=False)    
#     model = Sequential()    
#     model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))   
#     model.add(Dropout(0.2))    
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.summary()   
#     history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
#                         callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

#     model.save('lstm_model_scaled_data.h5')    
#     y_predict = model.predict(X_test)    
#     X_test = X_scaler.inverse_transform(X_test)
#     y_test = y_scaler.inverse_transform(y_test)
#     y_predict = y_scaler.inverse_transform(y_predict)    
#     plt.scatter(X_test, y_test, label='Actual', marker='o', alpha=0.7)
#     plt.scatter(X_test, y_predict, label='Predicted', marker='x', alpha=0.7)

    
#     next_cycle = max_cycle + 1
#     next_cycle_scaled = X_scaler.transform(np.array([[next_cycle]]))
#     next_cycle_prediction_scaled = model.predict(next_cycle_scaled)
#     next_cycle_prediction = y_scaler.inverse_transform(next_cycle_prediction_scaled)

#     plt.scatter(next_cycle, next_cycle_prediction, label='Next Predicted', marker='s', color='red', alpha=0.7)

#     plt.xlabel('Cycle')
#     plt.legend()
#     st.pyplot()  # Plot in column 1
    

#     plt.figure(figsize=(8, 4))
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Test Loss')
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epochs')
#     plt.legend(loc='upper right')
#     st.pyplot()  # Plot in column 2
    
#     return X_test, y_test, y_predict, history

     

    
    
    
    ##################Functions end

# Load your DataFrame here
df_cleaned = pd.read_csv('./data/df_cleaned.csv')
df_with_soh = pd.read_csv('data/df_with_soh.csv')
df_sorted = pd.read_csv('./data/sorted_file.csv')
df_discharge = pd.read_csv('./data/df_discharge.csv')
battery_options = ['B0005', 'B0006', 'B0028', 'B0029']
df_model = pd.read_csv('filtered_df_with_soh.csv')
battery_options2 = ['B0028', 'B0029', 'B0030']

def main():
    

    logo_url = './img/batt.jpg'
    st.sidebar.image(logo_url)

    # sidebar
    st.sidebar.title('Explore and Predict Battery Health')

    # Enhanced Sidebar Options
    view = st.sidebar.radio(
        'Choose an Option',
        ('Data Analysis 📊', 'Prediction with SVR Model','Prediction with LSTM Model' ),
        index=0
    )


    # Display the selected view as the main header
    if 'Data Analysis' in view:
        st.title('Data Analysis')
        
        
        col11, col12 = st.columns(2)
        with col11:
            st.subheader('Battery Capacity per Cycle Plot')
            # Battery selection
            selected_battery2 = st.selectbox('Select Battery Name', battery_options, index=0)

            # Plot
            plot_capacity_per_cycle(df_cleaned, selected_battery2)               
            
        with col12:
            st.subheader('Battery Capacity per Cycle Plot')
            selected_battery1 = st.selectbox('Select Battery Name', battery_options, index=2)

            # Plot
            plot_soh_per_cycle(df_with_soh, selected_battery1)
            
        # Custom dashed line or space
        st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
        
        col21, col22 = st.columns(2)
        with col21:
            st.markdown("""
            <div style='font-size: 18px;'><b>This dataset is from <a href='https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/'>NASA PCoE</a>.</b></div>
            """, unsafe_allow_html=True)
            st.write("""
            The dataset includes experiments on Li-Ion batteries. Charging and discharging were performed at different temperatures.
            The impedance was recorded as the damage criterion. The dataset was provided by the NASA Prognostics Center of Excellence (PCoE).
            """)
            
        
            
        with col22:       
                     
            # st.subheader('Battery **State of Health**')
            # st.latex(r'SOH = \frac{Q_{aged}}{Q_{rate}} \times 100')
            # st.markdown(r'where $Q_{rate}$ is the rated capacity of the battery when it leaves the factory.')
            
             st.markdown("""
                <div style="text-align: left"> 
                    <h4>Battery <b>State of Health</b></h4> 
                    <p>SOH (State of Health) is a measure of a battery's current condition compared to its original condition. 
                    The SOH is calculated as the ratio of the aged capacity of the battery to its rated capacity and then multiplied by 100.</p>
                    <p>In this case, we have considered the maximum capacity of each battery and calculated the SOH for each dataset within that particular battery.</p>
                    <img src="https://latex.codecogs.com/svg.latex?SOH%20%3D%20%5Cfrac%7BQ_%7Baged%7D%7D%7BQ_%7Brate%7D%7D%20%5Ctimes%20100" alt="SOH = \frac{Q_{aged}}{Q_{rate}} \times 100"/>
                    
                </div>
            """, unsafe_allow_html=True)
            
        ## line
        st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
        
        ######################  third row of data display
        

        # Create three columns
        # Create three columns with the middle one being the same width as a single column in a two-column layout
        col31, col32, col33 = st.columns([1,6,1])

        # Plot in the middle column
        with col32:
            st.subheader('Voltage Measured over Charge and Discharge Cycles')
        # Battery selection
            selected_battery3 = st.selectbox('Select Battery Number', ['B0005', 'B0006', 'B0028', 'B0029'], index=0)
            min_value1 = 0
            max_value1 = df_cleaned[df_cleaned['battery_name'] == selected_battery3]['id_cycle'].max()
            default_value1 = 15
            selected_cycles1=st.slider("Select a value", min_value1, max_value1, default_value1)
            plot_voltage_data(df_sorted, selected_battery3, selected_cycles1)
                
       
       
       # dashed line
        st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
        
        
        col41, col42, col43 = st.columns([1,6,1])
        
        # Plot in the middle column
        with col42:       
            
            # Use Streamlit to create a dropdown menu for battery selection
            selected_battery_name = st.selectbox("Select Battery Name for Temperature Analysis", battery_options)

            max_cycle42 = df_cleaned[df_cleaned['battery_name'] == selected_battery_name]['id_cycle'].max()
            st.write(f"The maximum id_cycle for {selected_battery_name} is: { max_cycle42}")
            default_value42 = 15
            
            max_value42 = df_sorted[df_sorted['battery_name'] == selected_battery_name]['id_cycle'].max()        
            selected_cycle42 = st.number_input("Select number of cycles", min_value=1, max_value=max_value42, value=18, step=1)

            # Plot for the selected battery and cycles
            plot_temperature_data(df_cleaned, selected_battery_name, selected_cycle42)
                
        # dashed line
        st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
        
        #####
        
        col51, col52, col53 = st.columns([1,5,1])
        with col52: 
            st.subheader('Battery voltage/current/temperature per Cycle Plot')
            selected_battery_name = st.selectbox("Select Battery Name for V/C Analysis", battery_options)

            max_cycle52 = df_cleaned[df_cleaned['battery_name'] == selected_battery_name]['id_cycle'].max()
            st.write(f"The maximum id_cycle for {selected_battery_name} is: { max_cycle42}")
            default_value42 = 15
            available_cycles = df_cleaned[df_cleaned['battery_name'] == selected_battery_name]['id_cycle'].unique()
            available_cycles_list = available_cycles.tolist()     
            # selected_cycle52 = st.number_input("Select the Cycle", min_value=1, max_value=max_value42, value=18, step=1)
            selected_cycle52 = st.selectbox("Select the Cycle", available_cycles_list)
        
        # row 5 of displaying
        col61, col62 = st.columns(2)
        
        # Plot in the middle column
        with col61:            
            plot_voltage_current_charge_over_time(df_sorted, selected_battery_name, selected_cycle52 )
            
        with col62:            
            plot_voltage_temperature_over_time(df_sorted, selected_battery_name, selected_cycle52 )
            
    elif 'SVR' in view:
        st.title('Data Analysis')        
        
        tab1, tab2 = st.tabs(["💡SVR Model for Available Batteries", "💡Enter Your Data for SVR Prediction"])
        

        with tab1:
            st.header('Machine Learning SVR Model')
            
            # Set the default selection to the first battery
            selected_battery_name = st.selectbox("Select Battery Name for Correlation Analysis", battery_options2, index=0)

            # Plot correlation heatmap for the selected battery
            fig, sorted_correlations = plot_correlation_heatmap(df_model, selected_battery_name)

        # Split the page into 2 columns
            col71, col72 = st.columns(2)        
            with col71:
                st.pyplot(fig)        
            with col72:            
                st.write('')  # Empty line
                st.write(f"The sorted correlations for the capacity value of battery {selected_battery_name} are presented below:")
                # Short solid red line
                # Short solid red line
                sorted_correlations_df = sorted_correlations.reset_index().rename(columns={'index': 'Feature', 'Capacity': 'Correlation'})            

                # Write the styled table centered in the column
                col81, col82, col83 = st.columns([1,5,1])
                with col82: 
                    st.dataframe(
                            sorted_correlations_df.style
                            .set_properties(**{'text-align': 'center'})
                        )
                    
                # dashed line
            st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
            
            
            selected_battery_name = st.selectbox("Select Battery for training and test", battery_options2, index=2)
            df_soh = pd.read_csv('filtered_df_with_soh.csv')            
            df_selected = df_soh[df_soh['battery_name']==selected_battery_name]
            X = df_selected['id_cycle'].values.reshape(-1, 1)
            Y = df_selected['SoH'].values.reshape(-1, 1)
            
            col91, col92, = st.columns(2)
            with col91: 
                scores = train_evaluate_plot(X, Y)
            with col92: 
                col911, col912, col913 = st.columns([1,5, 2])
                with col912:
                    st.write('')
                    st.write('')
                    for train_size, score in scores:
                        
                        st.info(f'The R2 score with train size of {train_size}%: {score:.2f}')
                        
            st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
            
            
            col1001, col1002 = st.columns(2)            
            with col1001:            
                selected_battery_name10 = st.selectbox("Select Battery ", battery_options2, index=2)
                df_soh = pd.read_csv('filtered_df_with_soh.csv')            
                df_selected = df_soh[df_soh['battery_name']==selected_battery_name10]
                X1 = df_selected['id_cycle'].values.reshape(-1, 1)
                Y1 = df_selected['SoH'].values.reshape(-1, 1)
            
            with col1002: 
                st.write('')                  
                
            col101, col102,col103, col104, col105 = st.columns(5)            
            with col101:    
                ratio = st.number_input('Enter the test size ratio (in %)', min_value=0, max_value=100, value=20)
                C = st.number_input('Enter the parameter C', min_value=0.0, value=20.0)
                
                  
                    
                    
            with col102:
                epsilon = st.number_input('Enter the epsilon', min_value=0.0, value=0.0001)              
                gamma = st.number_input('Enter the gamma', min_value=0.0, value=0.0001)
                
                
            with col103:
                cache_size = st.number_input('Enter the cache size', min_value=0, value=200)
                kernel = st.selectbox('Select the kernel', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
                
                
            with col104:
                max_iter = st.number_input('Enter the max iterations (-1 for no limit)', min_value=-1, value=-1)
                shrinking = st.checkbox('Use shrinking heuristic?', value=True)
                

            with col105:
                tol = st.number_input('Enter the tolerance for stopping criterion', min_value=0.0, value=0.001)
                verbose = st.checkbox('Enable verbose output?', value=False)  
            
            col111, col112, = st.columns(2)
            with col111:                
                scores_p = train_evaluate_plot_param(X1, Y1, ratio, C, epsilon, gamma, cache_size, kernel, max_iter, shrinking, tol, verbose)
                
                
            with col112: 
                col121, col122, col123 = st.columns([1,5, 2])
                with col122:
                    st.write('')
                    st.write('')
                    st.info(f'The R2 score with train size of {100 - ratio}%: {scores_p:.2f}')
           

        with tab2:
            st.write("Content for the second tab.")
            
    else:
        tab3, tab4 = st.tabs(["💡LSTM Model for Available Batteries", "💡Enter Your Data for LSTM Prediction"])
        

        with tab3:
            st.header('Deep Learning LSTM Model')
            
            # selected_battery_name_LSTM = st.selectbox("Select Battery for LSTM Model", battery_options2, index=2)
            # df_soh = pd.read_csv('filtered_df_with_soh.csv')            
            # df_selected = df_soh[df_soh['battery_name']==selected_battery_name_LSTM]
            # X = df_selected['id_cycle'].values.reshape(-1, 1)
            # Y = df_selected['SoH'].values.reshape(-1, 1)
            # max_cycle = df_selected[df_selected['battery_name'] == selected_battery_name_LSTM ]['id_cycle'].max()
            # train_evaluate_plot_lstm(X, Y, max_cycle)
            
                      
        
            
            
            
            # st.markdown("<hr style='border: 1px dashed #4285f4;'>", unsafe_allow_html=True) 
        


    
if __name__ == '__main__':
    main()
    