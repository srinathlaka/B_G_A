import streamlit as st
import plotly.graph_objects as go

# Function to plot Time vs OD with adjusted size
def plot_time_vs_od(df, well, plot_placeholder):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df[well], mode='lines', name=well))
    
    # Adjust the layout for a smaller, square-like plot
    fig.update_layout(
        title=f"Time vs {well} OD", 
        xaxis_title='Time', 
        yaxis_title='OD', 
        template='plotly_white',
        width=300,  # Set the width of the plot
        height=300  # Set the height to make it more square
    )
    plot_placeholder.plotly_chart(fig)

# Select wells using buttons and plot Time vs OD below the grid
def select_wells(df, rows, columns, labels, message, session_key):
    if session_key not in st.session_state:
        st.session_state[session_key] = set()

    selected_wells = sorted(list(st.session_state[session_key]))

    # Create a placeholder for the plots
    plot_container = st.empty()

    with st.container():
        for i in range(rows):
            cols = st.columns(columns)
            for j in range(columns):
                index = i * columns + j
                if index < len(labels):
                    button_label = labels[index]
                    if button_label != "Time":
                        button_key = f"button_{session_key}_{i}_{j}"
                        
                        # Check if the column associated with the well has data
                        if df[button_label].isnull().all():
                            # If the data is missing, disable the button
                            cols[j].button(button_label, key=button_key, disabled=True)
                        else:
                            # If data exists, create an active button
                            if cols[j].button(button_label, key=button_key):
                                if button_label in selected_wells:
                                    selected_wells.remove(button_label)
                                else:
                                    selected_wells.append(button_label)
                                    
                                    # Plot the selected well's data below the grid
                                    with plot_container.container():
                                        plot_time_vs_od(df, button_label, plot_container)
                                    
                                selected_wells = sorted(selected_wells)
                                st.session_state[session_key] = set(selected_wells)

    selected_wells_text = f"Currently Selected Wells: " + ", ".join(selected_wells) if selected_wells else "None"
    message.write(selected_wells_text)

    return selected_wells


# Clear selected wells
def clear_selected_wells(session_key, message):
    st.session_state[session_key].clear()
    message.warning("Selected wells cleared.")

# Perform background subtraction and set negative values to zero


def perform_background_subtraction(df, blank_wells, sample_wells):
    """
    Perform background subtraction by subtracting the average of the blank wells
    from the sample wells' data. This function also calculates the average and
    standard deviation for the selected sample wells.
    
    Parameters:
    - df: DataFrame containing the raw well data.
    - blank_wells: List of blank well names.
    - sample_wells: List of sample well names.
    
    Returns:
    - A new DataFrame with background-subtracted data, including 'Average' and 'Std_Dev' columns.
    """
    # Check if blank wells are provided
    if len(blank_wells) == 0:
        raise ValueError("No blank wells selected for background subtraction.")
    
    # Calculate the mean for the blank wells across all time points
    blank_avg = df[blank_wells].mean(axis=1)
    
    # Subtract the blank well average from each sample well
    df_subtracted = df.copy()
    for well in sample_wells:
        df_subtracted[well] = df[well] - blank_avg
        # Set negative values to zero
        df_subtracted[well] = df_subtracted[well].apply(lambda x: max(x, 0))
    
    # Calculate the average and standard deviation across the sample wells
    df_subtracted['Average'] = df_subtracted[sample_wells].mean(axis=1)
    df_subtracted['Std_Dev'] = df_subtracted[sample_wells].std(axis=1)
    
    return df_subtracted