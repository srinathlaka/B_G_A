import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import sympy as sp
import re
from utils.metrics import calculate_metrics
import plotly.express as px
import plotly.graph_objects as go

# User-provided ODE model class
class UserProvidedODE:
    def __init__(self, ode_function):
        self.ode_function = ode_function

    def model_fun(self, state, t, *params):
        return eval(self.ode_function)

# Function to solve the ODE
def solve_ode(model, t, *params):
    state0 = [0.01]  # Initial condition for the ODE
    sol = odeint(model.model_fun, state0, t, args=params)
    return sol[:, 0]

# Function to parse the user-provided ODE function
def parse_ode_function(ode_function):
    try:
        num_params = len(set([int(p[7:8]) for p in ode_function.split() if p.startswith("params[")]))
        return num_params
    except Exception as e:
        st.error(f"Invalid ODE function: {e}")
        return None

# Function to convert user ODE to Python format
def convert_to_python_format(ode_function):
    return ode_function.replace("^", "**")

# Function to display ODE in LaTeX
def display_ode_latex(ode_function):
    x = sp.symbols('x')
    params = sp.symbols('params0:10')  # Create 10 dummy parameters
    ode_sympy = sp.sympify(ode_function.replace('state[0]', 'x'), locals=dict(x=x, params=params))
    return sp.latex(ode_sympy)

# Function to display custom function in LaTeX
def display_custom_function_latex(custom_function):
    t = sp.symbols('t')
    # Parse the function string to get the symbols
    params = sorted(set(re.findall(r'[a-zA-Z]\w*', custom_function)) - {'t'})
    symbols = sp.symbols(' '.join(params))
    custom_sympy = sp.sympify(custom_function, locals=dict(t=t, **{param: sym for param, sym in zip(params, symbols)}))
    return sp.latex(custom_sympy)

# Initialize session state for phases
def initialize_session_state():
    if 'phases' not in st.session_state:
        st.session_state.phases = []
    for phase in st.session_state.phases:
        phase.setdefault('start', 0)
        phase.setdefault('end', 0)
        phase.setdefault('model', 'Exponential')
        phase.setdefault('initial_guesses', {})
        phase.setdefault('ode_function', '')
        phase.setdefault('custom_function', '')
    
    if 'submit_equation' not in st.session_state:
        st.session_state.submit_equation = False
    if 'num_params' not in st.session_state:
        st.session_state.num_params = 0
    if 'ode_function' not in st.session_state:
        st.session_state.ode_function = ''
    if 'ode_function_latex' not in st.session_state:
        st.session_state.ode_function_latex = ''
    if 'fit_results' not in st.session_state:
        st.session_state.fit_results = []

    
    # NEW PART - ensure df_bg_subtracted and plots are not reset
    if 'df_bg_subtracted' not in st.session_state:
        st.session_state.df_bg_subtracted = None
    if 'selected_wells_plot' not in st.session_state:
        st.session_state.selected_wells_plot = None
    if 'average_with_std_plot' not in st.session_state:
        st.session_state.average_with_std_plot = None

    # NEW: Ensure phase plots are stored and not reset
    if 'phase_plots' not in st.session_state:
        st.session_state.phase_plots = []  # List to store plots for each phase


def add_phase():
    st.session_state.phases.append({
        'start': 0, 
        'end': 0, 
        'model': 'Exponential', 
        'initial_guesses': {}, 
        'ode_function': '', 
        'custom_function': ''
    })

def delete_phase(phase_index):
    if phase_index < len(st.session_state.phases):
        # Remove the phase from the phases list
        st.session_state.phases.pop(phase_index)

        # Remove the corresponding fit result if it exists
        st.session_state.fit_results = [
            result for result in st.session_state.fit_results if result['phase_index'] != phase_index
        ]

        # Remove the corresponding plot for the deleted phase
        if 'phase_plots' in st.session_state and len(st.session_state.phase_plots) > phase_index:
            st.session_state.phase_plots.pop(phase_index)

        # Re-index fit results and plots after deletion
        for i, result in enumerate(st.session_state.fit_results):
            result['phase_index'] = i

        # Rerun the app to reflect the changes
        st.experimental_rerun()



def fit_model_to_phase(phase, phase_data):
    if phase['model'] == 'UserProvidedODE':
        return fit_user_provided_ode(phase, phase_data)
    elif phase['model'] == 'UserProvidedFunction':
        return fit_user_provided_function(phase, phase_data)
    else:
        # Return popt and pcov for predefined models
        return fit_predefined_model(phase, phase_data)


# Fit predefined growth models
def fit_predefined_model(phase, phase_data):
    from utils.growth_models import exponential_growth, logistic_growth, baranyi_growth, lag_exponential_saturation_growth
    
    if phase_data.empty:
        st.error("Phase data is empty. Please check the start and end times.")
        return None, None, None, None
    
    try:
        initial_value = phase_data['Average'].iloc[0]
    except IndexError:
        st.error("Phase data is empty or start/end times are incorrect. Please adjust the times.")
        return None, None, None, None

    models = {
        'Exponential': (exponential_growth, [0.0001, initial_value]),
        'Logistic': (logistic_growth, [0.0001, initial_value, 1.0]),
        'Baranyi': (baranyi_growth, [initial_value, 0.0001, 1.0]),
        'Lag-Exponential-Saturation': (lag_exponential_saturation_growth, [0.0001, initial_value, 1.0, 1.0])
    }
    
    model_func, initial_params = models[phase['model']]
    
    try:
        # Fitting the model and getting the covariance matrix (pcov)
        popt, pcov = curve_fit(model_func, phase_data['Time'], phase_data['Average'], p0=initial_params, maxfev=10000)
        
        # Calculate the fitted curve
        fit = model_func(phase_data['Time'], *popt)
        
        # Return model name, fitted curve, parameters, and covariance matrix
        return phase['model'], fit, popt, pcov
    except Exception as e:
        st.write(f"Error fitting {phase['model']} model: {e}")
        return None, None, None, None


def fit_user_provided_ode(phase, phase_data):
    model = UserProvidedODE(phase['ode_function'])
    initial_guess = [phase.get(f'param{j+1}', 0.1) for j in range(parse_ode_function(model.ode_function))]
    try:
        popt, _ = curve_fit(lambda t, *params: solve_ode(model, t, *params), phase_data['Time'], phase_data['Average'], p0=initial_guess)
        fit = solve_ode(model, phase_data['Time'], *popt)
        return phase['model'], fit, popt
    except Exception as e:
        st.write(f"Error fitting model: {e}")
        return None, None, None

def fit_user_provided_function(phase, phase_data):
    custom_function_str = phase['custom_function']
    params = re.findall(r'[a-zA-Z]\w*', custom_function_str)
    params = list(set(params) - {'t'})
    params.sort()

    if not params:
        st.error("No parameters found in the custom function. Please ensure your function includes at least one parameter.")
        return None, None, None

    lambda_str = f"lambda t, {', '.join(params)}: {custom_function_str}"
    custom_function = eval(lambda_str)
    initial_guess = [phase.get(param, 0.1) for param in params]
    try:
        popt, _ = curve_fit(custom_function, phase_data['Time'], phase_data['Average'], p0=initial_guess)
        fit = custom_function(phase_data['Time'], *popt)
        return phase['model'], fit, popt
    except Exception as e:
        st.write(f"Error fitting custom function: {e}")
        return None, None, None

# Function to create the plot
def create_plot(data):
    fig = px.line(data, x='Time', y='Average', markers=True, title='Time vs Average',
                  labels={'Time': 'Time', 'Average': 'Average'},
                  hover_data={'Time': True, 'Average': True})

    colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightgrey']
    for i, phase in enumerate(st.session_state.phases):
        color = colors[i % len(colors)]
        fig.add_vrect(x0=float(phase['start']), x1=float(phase['end']), fillcolor=color, opacity=0.3, line_width=0,
                      annotation_text=f"Phase {i + 1}", annotation_position="top left")
    
    valid_indices = set(range(len(st.session_state.phases)))
    st.session_state.fit_results = [result for result in st.session_state.fit_results if result['phase_index'] in valid_indices]
    
    for result in st.session_state.fit_results:
        phase_index = result["phase_index"]
        if phase_index < len(st.session_state.phases):
            phase_data = data[(data['Time'] > float(st.session_state.phases[phase_index]['start'])) & (data['Time'] <= float(st.session_state.phases[phase_index]['end']))]
            fig.add_trace(go.Scatter(x=phase_data['Time'], y=result["fit"], mode='lines', name=f'{result["model_name"]} Fit Phase {phase_index + 1}'))

    return fig

# Calculate Exponential Moving Average
def exponential_moving_average(data, span):
    return data.ewm(span=span, adjust=False).mean()

# Detect phases
def detect_phases(time, average, distance_value, prominence_value, span=5):
    average_ema = exponential_moving_average(average, span)
    peaks_ema, _ = signal.find_peaks(average_ema, prominence=prominence_value, distance=distance_value)
    derivative_ema = np.diff(average_ema)
    change_points_ema = signal.find_peaks(np.abs(derivative_ema), prominence=prominence_value, distance=distance_value)[0]
    return average_ema, peaks_ema, change_points_ema

# Function to plot detected phases
def plot_detected_phases(time, average_ema, peaks_ema, change_points_ema):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=average_ema, mode='lines', name='EMA Average'))
    fig.add_trace(go.Scatter(x=time[peaks_ema], y=average_ema[peaks_ema], mode='markers', marker=dict(color='red'), name='Local Maxima'))
    fig.add_trace(go.Scatter(x=time[1:][change_points_ema], y=average_ema[1:][change_points_ema], mode='markers', marker=dict(color='green'), name='Change Points'))
    return fig


def plot_phase_fit_with_ci(full_data, phase_time, fit, lower_bound, upper_bound, std_dev, phase_model):
    fig = go.Figure()

    # Plot the original background-subtracted data
    fig.add_trace(go.Scatter(
        x=full_data['Time'],
        y=full_data['Average'],
        mode='lines',
        name='Original Data',
        line=dict(color='black', dash='dash')
    ))

    # Plot the fit curve
    fig.add_trace(go.Scatter(
        x=phase_time,
        y=fit,
        mode='lines',
        name=f'{phase_model} Fit',
        line=dict(color='blue', width=2)
    ))

    # Plot the confidence intervals
    fig.add_trace(go.Scatter(
        x=phase_time,
        y=lower_bound,
        fill=None,
        mode='lines',
        line_color='lightgrey',
        name='Lower CI'
    ))

    fig.add_trace(go.Scatter(
        x=phase_time,
        y=upper_bound,
        fill='tonexty',  # Fill to the previous trace (lower bound)
        mode='lines',
        line_color='lightgrey',
        name='Upper CI'
    ))

    # Plot standard deviation as shaded area (or error bars, depending on your preference)
    fig.add_trace(go.Scatter(
        x=phase_time,
        y=fit + std_dev,
        fill=None,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Fit + SD'
    ))

    fig.add_trace(go.Scatter(
        x=phase_time,
        y=fit - std_dev,
        fill='tonexty',
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Fit - SD'
    ))

    # Update layout with titles and labels
    fig.update_layout(
        title=f'Phase Fit with CI and SD ({phase_model} Model)',
        xaxis_title='Time',
        yaxis_title='Optical Density (OD)',
        template='plotly_white'
    )

    return fig


import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_all_fits(full_data, fit_results):
    fig = go.Figure()

    # Plot the original data (Time vs Average) first
    fig.add_trace(go.Scatter(
        x=full_data['Time'],
        y=full_data['Average'],
        mode='lines',
        name='Original Data',
        line=dict(color='black', dash='dash')
    ))

    # Dynamically add fit results after plotting the original data
    for result in fit_results:
        phase_index = result["phase_index"]
        fit_name = f"Fit Attempt {phase_index + 1}"

        # Add the fitted curve for each phase
        fig.add_trace(go.Scatter(
            x=full_data['Time'],
            y=result['fit'],
            mode='lines',
            name=fit_name,
            line=dict(width=2)
        ))

    # Customize the layout with titles and labels
    fig.update_layout(
        title='All Fit Curves',
        xaxis_title='Time',
        yaxis_title='Optical Density (OD)',
        template='plotly_white',
        legend_title='Fits'
    )

    return fig



