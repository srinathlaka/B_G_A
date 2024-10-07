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

# Perform background subtraction
def perform_background_subtraction(df, selected_blank_wells, selected_sample_replicates):
    if len(selected_blank_wells) == 0 or len(selected_sample_replicates) == 0:
        st.warning("Please select both blank wells and sample replicates for subtraction.")
        return None

    selected_blank_wells_list = list(selected_blank_wells)
    blank_mean = df[selected_blank_wells_list].mean(axis=1)

    for sample_replicate in selected_sample_replicates:
        df[sample_replicate] = df[sample_replicate] - blank_mean

    return df