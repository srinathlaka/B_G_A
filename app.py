import re
import streamlit as st
import numpy as np
import pandas as pd
from components.file_upload import upload_file, read_data, select_layout, generate_labels
from components.data_processing import select_wells, clear_selected_wells, perform_background_subtraction
from components.model_fitting import fit_growth_model, compute_confidence_intervals
from components.visualization import plot_avg_and_std, plot_confidence_intervals, plot_selected_wells,plot_time_vs_average_with_std
from components.phase_analysis import convert_to_python_format,plot_all_fits, display_custom_function_latex, display_ode_latex, initialize_session_state, create_plot, plot_phase_fit_with_ci, parse_ode_function, plot_detected_phases, add_phase, delete_phase, fit_model_to_phase
from utils.growth_models import polynomial_growth, polynomial_func
from utils.metrics import calculate_metrics
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Bacterial Growth Analysis", page_icon="🔬", layout="wide")

def create_sample_file():
    sample_data = {
        "Time": [0, 1, 2, 3, 4],
        "A1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "A2": [0.15, 0.25, 0.35, 0.45, 0.55],
        "B1": [0.05, 0.1, 0.15, 0.2, 0.25],
        "B2": [0.2, 0.3, 0.4, 0.5, 0.6]
    }
    df = pd.DataFrame(sample_data)
    return df

def show_sample_file_section():
    st.subheader("Sample Data Format")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])  # col1 for the example df, col2 for the image

    with col1:
        st.write("""
            Please ensure that your file follows this format:
            
            - The first column should be labeled `Time` and contain the time points.
            - Subsequent columns should represent well measurements, like `A1`, `A2`, etc.
        """)

        # Display the example DataFrame in the first column
        sample_file = create_sample_file()
        st.dataframe(sample_file)

    with col2:
        # Display the image in the second column
        st.image("assets/image.png", caption="96-Well Plate Layout", use_column_width=True)


def provide_example_excel():
    with open("assets/example_spreadsheet.xlsx", "rb") as file:
        st.download_button(
            label="Download Example Spreadsheet Fo Reference",
            data=file,
            file_name="example_spreadsheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    st.title("Growth Curve Fitting")
    st.write("This app is designed to fit growth curves to your experimental data.")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bacterial Growth Analysis</h1>", unsafe_allow_html=True)
    st.write("The app allows you to upload your bacterial growth data, fit growth models, perform background subtraction, and visualize the results.")
    st.write("Please follow the instructions below to analyze your data.")
    if 'fit_results' not in st.session_state:
        st.session_state['fit_results'] = []

    with st.expander("Instructions for Using the Bacterial Growth Analysis Tool"):
        st.markdown("""
        1. **Upload Your Data**:
           - Click on "Choose a file" to upload your dataset in .xlsx or .csv format. Ensure your data includes a 'Time' column and OD measurements for each well.

        2. **Select Well Layout**:
           - Choose a pre-defined well layout from the dropdown or enter custom dimensions for rows and columns.

        3. **Select Wells for Analysis**:
           - Use the generated buttons to select wells for blanks and samples. Selected wells will be highlighted.

        4. **Fit Growth Model**:
           - Choose the desired growth model (Polynomial Growth or Polynomial Function) and fit the model to your data. View the fitted curve and confidence intervals.

        5. **Background Subtraction**:
           - Perform background subtraction by selecting blank wells and sample replicates. The adjusted data will be displayed for further analysis.

        6. **Visualize Results**:
           - View various plots including the observed data, fitted curves, average measurements, and standard deviations. Toggle options to customize the display.
        """)

    st.sidebar.header("Bacterial Growth Analysis")
    st.sidebar.write("Please upload the files in .xlsx or .csv format only")

    # Provide a link to download the example spreadsheet
    st.subheader("Spreadsheet Example")
    st.write("You can download the example spreadsheet file with the OD table and plate layout.")
    try:
        provide_example_excel()
    except FileNotFoundError:
        st.error("Example Excel file not found.")

    # Show the sample file format to guide users
    show_sample_file_section()

    try:
        rows, columns = select_layout()
    except Exception as e:
        st.error("Error selecting layout. Please check your input.")

    uploaded_file = upload_file()



    if uploaded_file is not None:
        df = read_data(uploaded_file, rows, columns)
        if df is not None:
            try:
                labels = generate_labels(rows, columns)
                st.write(f"Grid dimensions: {rows} rows x {columns} columns")
                st.write(df)
            except ValueError as ve:
                st.error(f"Error: {ve}")
                st.write(f"Length of labels list: {len(labels)}")
                st.write(f"Number of columns in DataFrame: {df.shape[1]}")
                return

            st.subheader("Select Blank Wells")
            selected_blank_wells_message = st.empty()
            selected_blank_wells = select_wells(df, rows, columns, labels, selected_blank_wells_message, "selected_blank_wells")
            std_dev_blank_cells = df[selected_blank_wells].std(axis=1)
            df['Average'] = df[selected_blank_wells].mean(axis=1)

            # Store model selection in session state
            if 'selected_model' not in st.session_state:
                st.session_state['selected_model'] = "Polynomial Growth"
            
            selected_model = st.selectbox("Select Growth Model", ["Polynomial Growth", "Polynomial Function"], index=["Polynomial Growth", "Polynomial Function"].index(st.session_state['selected_model']))
            st.session_state['selected_model'] = selected_model

            # Initialize session state for fit parameters
            if 'popt' not in st.session_state:
                st.session_state['popt'] = None
            if 'pcov' not in st.session_state:
                st.session_state['pcov'] = None

            # Fit model and store result in session state
            if st.button("Fit Model to Blank Wells"):
                try:
                    # Fit the model and get parameters, errors, and covariance matrix
                    popt, perr, pcov = fit_growth_model(selected_model, df['Time'], df['Average'])
                    st.session_state['popt'] = popt
                    st.session_state['pcov'] = pcov
                    
                    # Display the fitted parameters
                    st.write("Fitted Parameters:", popt)
                    
                    # Display the parameter errors (standard deviations)
                    if perr is not None:
                        st.write("Parameter Errors (Standard Deviations):", perr)
                    else:
                        st.write("Could not calculate parameter errors.")
                    
                    # Compute the fitted curve based on the model
                    if selected_model == "Polynomial Growth":
                        y_pred = polynomial_growth(df['Time'], *popt)
                    elif selected_model == "Polynomial Function":
                        y_pred = polynomial_func(df['Time'], *popt)

                    # Plot average measurements and standard deviation first
                    st.session_state['fitted_plot'] = plot_avg_and_std(df, selected_blank_wells, y_pred, std_dev_blank_cells)

                    # Compute residual variance and degrees of freedom
                    residuals = df['Average'] - y_pred
                    dof = len(df['Time']) - len(popt)
                    residual_variance = np.var(residuals, ddof=dof)

                    # Compute confidence intervals
                    lower_bound, upper_bound = compute_confidence_intervals(
                        df['Time'], popt, pcov, 0.05, dof, residual_variance, selected_model
                    )

                    st.session_state['ci_plot'] = plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev_blank_cells)

                except Exception as e:
                    st.error(f"Error during fitting: {e}")




            # Display the stored blank wells plot if available
            if 'fitted_plot' in st.session_state and st.session_state['fitted_plot'] is not None:
                st.plotly_chart(st.session_state['fitted_plot'])

            # Display the stored confidence interval plot if available
            if 'ci_plot' in st.session_state and st.session_state['ci_plot'] is not None:
                st.plotly_chart(st.session_state['ci_plot'])



            clear_blank_wells_button = st.button("Clear selected blank wells")
            if clear_blank_wells_button:
                clear_selected_wells("selected_blank_wells", selected_blank_wells_message)

            st.subheader("Select Sample Replicates")
            selected_sample_replicates_message = st.empty()
            selected_sample_replicates = select_wells(df, rows, columns, labels, selected_sample_replicates_message, "selected_sample_replicates")

            clear_sample_replicates_button = st.button("Clear selected sample replicates")
            if clear_sample_replicates_button:
                clear_selected_wells("selected_sample_replicates", selected_sample_replicates_message)

            if st.button("Perform Background Subtraction"):
                try:
                    st.session_state.df_bg_subtracted = perform_background_subtraction(df.copy(), selected_blank_wells, selected_sample_replicates)
                    st.write("Background Subtracted Data:")
                    st.dataframe(st.session_state.df_bg_subtracted)  # Show DataFrame with Average and Std_Dev
                except Exception as e:
                    st.error("Error performing background subtraction. Please ensure correct well selection.")

                if st.session_state.df_bg_subtracted is not None:
                    # Plot the individual selected wells
                    st.session_state['selected_wells_plot'] = plot_selected_wells(st.session_state.df_bg_subtracted, selected_sample_replicates)
                                    
                    # Plot the Average with Std Dev
                    st.session_state['average_with_std_plot'] = plot_time_vs_average_with_std(st.session_state.df_bg_subtracted)

    
    # Define a placeholder for the plot
    plot_placeholder = st.empty()  # Create the placeholder here

    if "df_bg_subtracted" in st.session_state:
        df_bg_subtracted = st.session_state.df_bg_subtracted
        initialize_session_state()

        # Define a placeholder for the background-subtracted plot
        if 'bg_subtracted_plot' not in st.session_state:
            st.session_state['bg_subtracted_plot'] = None  # Initialize plot storage

        # Plot the background-subtracted data if it's not already stored
        if st.session_state['bg_subtracted_plot'] is None:
            fig = create_plot(df_bg_subtracted)
            st.session_state['bg_subtracted_plot'] = fig
        else:
            fig = st.session_state['bg_subtracted_plot']

        # Display the background-subtracted plot
        st.subheader("Background Subtracted Plot")
        st.plotly_chart(fig)

        # Clear Button for background-subtracted plot
        if st.button('Clear Background Subtracted Plot'):
            st.session_state['bg_subtracted_plot'] = None  # Clear only the background-subtracted plot
            st.experimental_rerun()  # Re-run the app to remove the plot from the UI

        # Ensure fit_results is initialized if it doesn't exist
        if 'fit_results' not in st.session_state:
            st.session_state['fit_results'] = []

        # Check if fit results are available and display the combined plot of all fits
        if len(st.session_state['fit_results']) > 0:
            # Call the function to plot all fits in one combined plot
            #all_fits_plot = plot_all_fits(st.session_state.df_bg_subtracted, st.session_state.fit_results)

            # Display the plot at the top of the Phase Analysis section
            #st.subheader("All Fit Attempts in One Plot")
            #st.plotly_chart(all_fits_plot)

            # Clear Button for the combined fit plot
            if st.button('Clear All Fit Plot'):
                st.session_state['all_fits_plot'] = None  # Clear the combined fit plot
                st.experimental_rerun()  # Re-run the app to remove the plot from the UI


        # Button to clear plot
        #if st.button('Clear Plot'):
            # Clear the stored plot in session state
            #st.session_state['bg_subtracted_plot'] = None
            #st.session_state.fit_results = []  # Clear manual fit results
            #st.session_state.phases = []  # Clear any phase data used for automatic fitting
            #st.experimental_rerun()  # Force rerun the app to reflect changes


        # If the user selects "Manual" phase detection
        st.button('Add Phase', on_click=add_phase)

        for i, phase in enumerate(st.session_state.phases):
            with st.expander(f"Phase {i + 1}"):
                col1, col2 = st.columns(2)
                with col1:
                    phase['start'] = st.text_input(f'Start Time for Phase {i + 1}', value=str(phase['start']), key=f'start_{i}')
                with col2:
                    phase['end'] = st.text_input(f'End Time for Phase {i + 1}', value=str(phase['end']), key=f'end_{i}')

                # Delete button for the current phase
                if st.button(f'Delete Phase {i + 1}', key=f'delete_{i}'):
                    delete_phase(i)
                    st.experimental_rerun()
                        
                phase['model'] = st.selectbox(f'Model', ['Exponential', 'Logistic', 'Baranyi', 'Lag-Exponential-Saturation', 'UserProvidedODE', 'UserProvidedFunction'], index=0, key=f'model_{i}')

                phase_data = df_bg_subtracted[(df_bg_subtracted['Time'] > float(phase['start'])) & (df_bg_subtracted['Time'] <= float(phase['end']))]
                #st.write(f"Phase Data for phase {i + 1}:", phase_data)


                if phase['model'] == 'UserProvidedODE' and not phase['automatic']:
                    ode_function_input = st.text_area(f'ODE Function for Phase {i + 1}', value=phase['ode_function'], key=f'ode_function_{i}')
                    st.write("Hint: Write the ODE in terms of state and params, e.g., 'params[0] * state[0] * (1 - state[0] / params[1])'")
                    st.write("""
                        ### Examples:
                        - Logistic growth: `params[0] * state[0] * (1 - state[0] / params[1])`
                        - Exponential growth: `params[0] * state[0]`
                        """)

                    if st.button('Submit Equation', key=f'submit_{i}'):
                        st.session_state.phases[i]['ode_function'] = convert_to_python_format(ode_function_input)
                        st.session_state.num_params = parse_ode_function(st.session_state.phases[i]['ode_function'])
                        st.session_state.ode_function_latex = display_ode_latex(ode_function_input)
                        st.session_state.submit_equation = True

                    if st.session_state.submit_equation and st.session_state.phases[i]['ode_function']:
                        num_params = st.session_state.num_params
                        st.latex(st.session_state.ode_function_latex)
                        for param in range(num_params):
                            phase[f'param{param+1}'] = st.number_input(f'Parameter {param+1} for Phase {i + 1}', value=0.1, key=f'param{param+1}_{i}')
                        if st.button('Fit ODE', key=f'fit_{i}'):
                            model_name, fit, popt = fit_model_to_phase(phase, phase_data)
                            if model_name:
                                metrics = calculate_metrics(phase_data['Average'], fit, len(popt))
                                st.session_state.fit_results.append({
                                    "model_name": model_name,
                                    "fit": fit,
                                    "popt": popt,
                                    "metrics": metrics,
                                    "phase_index": i
                                })
                                fig = create_plot(df_bg_subtracted)
                                plot_placeholder.plotly_chart(fig)

                elif phase['model'] == 'UserProvidedFunction' and not phase['automatic']:
                    custom_function_input = st.text_area(f'Custom Function for Phase {i + 1}', value=phase['custom_function'], key=f'custom_function_{i}')
                    st.write("Hint: Write the custom function in terms of t and parameters (e.g., 'a*t**2 + b*t + c')")
                    st.write("""
                        ### Examples:
                        - Linear growth: `a*t + b`
                        - Quadratic growth: `a*t**2 + b*t + c`
                        """)

                    if st.button('Submit Custom Function', key=f'submit_custom_{i}'):
                        st.session_state.phases[i]['custom_function'] = custom_function_input
                        st.session_state.num_params = len(re.findall(r'[a-zA-Z]\w*', custom_function_input)) - 1
                        st.session_state.submit_equation = True

                    if st.session_state.submit_equation and st.session_state.phases[i]['custom_function']:
                        st.latex(display_custom_function_latex(st.session_state.phases[i]['custom_function']))
                        params = re.findall(r'[a-zA-Z]\w*', st.session_state.phases[i]['custom_function'])
                        params = list(set(params) - {'t'})
                        params.sort()
                        for param in params:
                            phase[param] = st.number_input(f'Parameter {param} for Phase {i + 1}', value=0.1, key=f'param_custom_{param}_{i}')
                        if st.button('Fit Custom Function', key=f'fit_custom_{i}'):
                            model_name, fit, popt = fit_model_to_phase(phase, phase_data)
                            if model_name:
                                metrics = calculate_metrics(phase_data['Average'], fit, len(popt))
                                st.session_state.fit_results.append({
                                    "model_name": model_name,
                                    "fit": fit,
                                    "popt": popt,
                                    "metrics": metrics,
                                    "phase_index": i
                                })
                                fig = create_plot(df_bg_subtracted)
                                plot_placeholder.plotly_chart(fig)

                else:
                    st.write(f"### Initial Guesses for {phase['model']} Model")
                    initial_guesses = {}
                    if not phase_data.empty:
                        if phase['model'] == 'Exponential':
                            cols = st.columns(2)
                            initial_guesses['mu'] = cols[0].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')
                            initial_guesses['X0'] = cols[1].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                        elif phase['model'] == 'Logistic':
                            cols = st.columns(3)
                            initial_guesses['mu'] = cols[0].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')
                            initial_guesses['X0'] = cols[1].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                            initial_guesses['K'] = cols[2].number_input('Initial guess for K', value=1.0, key=f'K_{i}')
                        elif phase['model'] == 'Baranyi':
                            cols = st.columns(3)
                            initial_guesses['mu'] = cols[0].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')  # Change mu to the first position
                            initial_guesses['X0'] = cols[1].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                            initial_guesses['q0'] = cols[2].number_input('Initial guess for q0', value=1.0, key=f'q0_{i}')

                        elif phase['model'] == 'Lag-Exponential-Saturation':
                            cols = st.columns(4)
                            initial_guesses['mu'] = cols[0].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')
                            initial_guesses['X0'] = cols[1].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                            initial_guesses['q0'] = cols[2].number_input('Initial guess for q0', value=1.0, key=f'q0_{i}')
                            initial_guesses['K'] = cols[3].number_input('Initial guess for K', value=1.0, key=f'K_{i}')
                        phase['initial_guesses'] = initial_guesses

                        # Replace create_plot() with your new plotting function
                        if st.button('Fit Model', key=f'fit_{i}'):
                            model_name, fit, popt, pcov = fit_model_to_phase(phase, phase_data)
                            if model_name:
                                # Calculate residual variance and degrees of freedom
                                residuals = phase_data['Average'] - fit
                                dof = len(phase_data['Time']) - len(popt)
                                residual_variance = np.var(residuals, ddof=dof)
                                # Compute standard deviation (SD) for the phase
                                std_dev = phase_data['Average'].std()
                                
                                metrics = calculate_metrics(phase_data['Average'], fit, len(popt))
                                st.write("Appending fit results to session state")
                                st.session_state.fit_results.append({
                                    "model_name": model_name,
                                    "fit": fit,
                                    "popt": popt,
                                    "metrics": calculate_metrics(phase_data['Average'], fit, len(popt)),
                                    "phase_index": i
                                })
                                #st.write("Fit results after appending:", st.session_state.fit_results)
                                #standard errors
                                perr = np.sqrt(np.diag(pcov))
                                
                                st.write("Parameter Errors (Standard Deviations):", perr)
                                
                                # Compute confidence intervals if covariance matrix is available
                                if pcov is not None:
                                    lower_bound, upper_bound = compute_confidence_intervals(
                                        phase_data['Time'], popt, pcov, 0.05, dof, residual_variance, phase['model']
                                    )

                                    # Use your new plotting function to plot with confidence intervals
                                    fig = plot_phase_fit_with_ci(
                                        full_data=df_bg_subtracted,      # Full background-subtracted data
                                        phase_time=phase_data['Time'],   # Time for the phase
                                        fit=fit,                         # Fitted data for the phase
                                        lower_bound=lower_bound,         # Confidence interval lower bound
                                        upper_bound=upper_bound,         # Confidence interval upper bound
                                        std_dev=std_dev,                 # sd
                                        phase_model=phase['model']       # Model name for the phase
                                    )

                                # Store the plot in session state
                                if len(st.session_state.phase_plots) <= i:
                                    st.session_state.phase_plots.append(fig)
                                else:
                                    st.session_state.phase_plots[i] = fig  # Update the plot for this phase

                                    # Display the plot
                                    st.plotly_chart(fig)
                                    st.write("it working")

                        # Display all phase plots after fitting
                        
                        for i, plot in enumerate(st.session_state.phase_plots):
                            st.plotly_chart(plot, use_container_width=True)


        # Display fit results for each phase
        #st.write("Fit Results Content:", st.session_state.fit_results)

        for result in st.session_state.fit_results:
            # Dynamically assign parameter names based on the model
            if result['model_name'] == 'Exponential':
                params = ['mu', 'X0']
            elif result['model_name'] == 'Logistic':
                params = ['mu', 'X0', 'K']
            elif result['model_name'] == 'Baranyi':
                params = ['mu', 'X0', 'q0']
            elif result['model_name'] == 'Lag-Exponential-Saturation':
                params = ['mu', 'X0', 'q0', 'K']
            else:
                params = [f'Parameter {i+1}' for i in range(len(result["popt"]))]

            # Parameters DataFrame for fitted values
            params_df = pd.DataFrame([result["popt"]], columns=params)
            st.write(f"### Fitted Parameters for Fit Attempt {result['phase_index'] + 1}")
            st.table(params_df)

            # Metrics DataFrame for RSS, R-squared, and AIC
            metrics_df = pd.DataFrame({
                "Model": [result["model_name"]],
                "RSS": [result["metrics"][0]],
                "R-squared": [result["metrics"][1]],
                "AIC": [result["metrics"][2]],
            })
            st.table(metrics_df)






if __name__ == "__main__":
    main()

