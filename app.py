import re
import streamlit as st
import numpy as np
import pandas as pd
from components.file_upload import upload_file, read_data, select_layout, generate_labels
from components.data_processing import select_wells, clear_selected_wells, perform_background_subtraction
from components.model_fitting import fit_growth_model, compute_confidence_intervals
from components.visualization import plot_avg_and_std, plot_confidence_intervals, plot_selected_wells
from components.phase_analysis import convert_to_python_format, display_custom_function_latex, display_ode_latex, initialize_session_state, create_plot, detect_phases, parse_ode_function, plot_detected_phases, add_phase, delete_phase, fit_model_to_phase
from utils.growth_models import polynomial_growth, polynomial_func
from utils.metrics import calculate_metrics

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
    st.write("""
        Please ensure that your file follows this format:
        
        - The first column should be labeled `Time` and contain the time points.
        - Subsequent columns should represent well measurements, like `A1`, `A2`, etc.
    """)

    # Display a small sample DataFrame as an example
    sample_file = create_sample_file()
    st.dataframe(sample_file)

def provide_example_excel():
    with open("assets/test1.xlsx", "rb") as file:
        st.download_button(
            label="Download Example Spreadsheet",
            data=file,
            file_name="example_spreadsheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    st.set_page_config(page_title="Bacterial Growth Analysis", page_icon="🔬", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bacterial Growth Analysis</h1>", unsafe_allow_html=True)

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
                    popt, pcov = fit_growth_model(selected_model, df['Time'], df['Average'])
                    st.session_state['popt'] = popt
                    st.session_state['pcov'] = pcov
                    st.write("Fitted Parameters:", popt)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")

            # Store and display the plot in session state
            if 'fitted_plot' not in st.session_state:
                st.session_state['fitted_plot'] = None

            if st.session_state['popt'] is not None:
                if selected_model == "Polynomial Growth":
                    y_pred = polynomial_growth(df['Time'], *st.session_state['popt'])
                elif selected_model == "Polynomial Function":
                    y_pred = polynomial_func(df['Time'], *st.session_state['popt'])

                st.session_state['fitted_plot'] = plot_avg_and_std(df, selected_blank_wells, y_pred, std_dev_blank_cells)

            # Display plot if it exists
            if st.session_state['fitted_plot'] is not None:
                st.plotly_chart(st.session_state['fitted_plot'])

            # Display confidence intervals
            if st.session_state['popt'] is not None and st.session_state['pcov'] is not None:
                residuals = df['Average'] - y_pred
                dof = len(df['Time']) - len(st.session_state['popt'])
                residual_variance = np.var(residuals, ddof=dof)

                # Use the correct model type for computing confidence intervals
                lower_bound, upper_bound = compute_confidence_intervals(
                    df['Time'], st.session_state['popt'], st.session_state['pcov'], 0.05, dof, residual_variance, selected_model
                )

                plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev_blank_cells)

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
                except Exception as e:
                    st.error("Error performing background subtraction. Please ensure correct well selection.")

                if st.session_state.df_bg_subtracted is not None:
                    st.success("Background subtraction completed successfully!")
                    st.write(st.session_state.df_bg_subtracted)
                    plot_selected_wells(st.session_state.df_bg_subtracted, selected_sample_replicates)
                    st.session_state.df_bg_subtracted['Average'] = st.session_state.df_bg_subtracted[selected_sample_replicates].mean(axis=1)

    if "df_bg_subtracted" in st.session_state:
        df_bg_subtracted = st.session_state.df_bg_subtracted
        initialize_session_state()

        if st.button('Clear Plot'):
            st.session_state.fit_results = []
            st.experimental_rerun()

        use_auto_detection = st.checkbox("Use Automatic Phase Detection", value=True)

        try:
            fig = create_plot(df_bg_subtracted)
            st.plotly_chart(fig)
        except Exception as e:
            st.error("Error generating plot. Please check your data.")

        plot_placeholder = st.plotly_chart(fig)

        if use_auto_detection:
            st.write("### Phase Detection Parameters")
            distance_value = st.number_input("Distance Value", value=50, min_value=1)
            prominence_value = st.number_input("Prominence Value", value=0.002, min_value=0.0001, step=0.0001)

            if st.button('Find Phases'):
                average_ema, peaks_ema, change_points_ema = detect_phases(df_bg_subtracted['Time'], df_bg_subtracted['Average'], distance_value, prominence_value)
                fig = plot_detected_phases(df_bg_subtracted['Time'], average_ema, peaks_ema, change_points_ema)
                st.plotly_chart(fig)

                # Append phases to session state
                st.session_state.phases = []
                for start, end in zip(change_points_ema[:-1], change_points_ema[1:]):
                    st.session_state.phases.append({
                        'start': df_bg_subtracted['Time'].iloc[start],
                        'end': df_bg_subtracted['Time'].iloc[end],
                        'model': 'Exponential',
                        'automatic': False,
                        'initial_guesses': {},
                        'ode_function': '',
                        'custom_function': ''
                    })
        else:
            st.button('Add Phase', on_click=add_phase)

            for i, phase in enumerate(st.session_state.phases):
                with st.expander(f"Phase {i + 1}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        phase['start'] = st.text_input(f'Start Time for Phase {i + 1}', value=str(phase['start']), key=f'start_{i}')
                    with col2:
                        phase['end'] = st.text_input(f'End Time for Phase {i + 1}', value=str(phase['end']), key=f'end_{i}')
                    phase['automatic'] = st.checkbox(f'Automatic Fit', value=phase['automatic'], key=f'automatic_{i}')
                    if not phase['automatic']:
                        phase['model'] = st.selectbox(f'Model', ['Exponential', 'Logistic', 'Baranyi', 'Lag-Exponential-Saturation', 'UserProvidedODE', 'UserProvidedFunction'], index=0, key=f'model_{i}')

                    phase_data = df_bg_subtracted[(df_bg_subtracted['Time'] > float(phase['start'])) & (df_bg_subtracted['Time'] <= float(phase['end']))]

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

                    elif not phase['automatic']:
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
                                initial_guesses['X0'] = cols[0].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                                initial_guesses['mu'] = cols[1].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')
                                initial_guesses['q0'] = cols[2].number_input('Initial guess for q0', value=1.0, key=f'q0_{i}')
                            elif phase['model'] == 'Lag-Exponential-Saturation':
                                cols = st.columns(4)
                                initial_guesses['mu'] = cols[0].number_input('Initial guess for mu', value=0.0001, key=f'mu_{i}')
                                initial_guesses['X0'] = cols[1].number_input('Initial guess for X0', value=phase_data['Average'].iloc[0], key=f'X0_{i}')
                                initial_guesses['q0'] = cols[2].number_input('Initial guess for q0', value=1.0, key=f'q0_{i}')
                                initial_guesses['K'] = cols[3].number_input('Initial guess for K', value=1.0, key=f'K_{i}')
                            phase['initial_guesses'] = initial_guesses

                            if st.button('Fit Model', key=f'fit_{i}'):
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

                    if phase['automatic']:
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

                    if st.button('Delete Phase', key=f'delete_{i}'):
                        delete_phase(i)
                        st.experimental_rerun()

        # Display fit results for each phase
        for result in st.session_state.fit_results:
            st.write(f"### Fit Results for Phase {result['phase_index'] + 1}")
            metrics_df = pd.DataFrame({
                "Model": [result["model_name"]],
                "RSS": [result["metrics"][0]],
                "R-squared": [result["metrics"][1]],
                "AIC": [result["metrics"][2]]
            })
            st.write(metrics_df)
            params_df = pd.DataFrame([result["popt"]], columns=['Parameter ' + str(i+1) for i in range(len(result["popt"]))])
            st.write(f"### Fitted Parameters for Phase {result['phase_index'] + 1}")
            st.write(params_df)

if __name__ == "__main__":
    main()

