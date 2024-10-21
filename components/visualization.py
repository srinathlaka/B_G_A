import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Plot average and standard deviation with Plotly
def plot_avg_and_std(df, selected_wells, y_pred=None, std_dev=None, log_scale=False):
    if len(selected_wells) > 0:
        selected_wells_list = list(selected_wells)
        df['Average'] = df[selected_wells_list].mean(axis=1)
        df['Std Dev'] = df[selected_wells_list].std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Average'], mode='lines', name='Average Measurement'))

        fig.add_trace(go.Scatter(
            x=df['Time'].tolist() + df['Time'].tolist()[::-1],
            y=(df['Average'] - df['Std Dev']).tolist() + (df['Average'] + df['Std Dev']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Standard Deviation'
        ))

        if y_pred is not None:
            fig.add_trace(go.Scatter(x=df['Time'], y=y_pred, mode='lines', name='Fitted Curve', line=dict(color='red')))

        fig.update_layout(
            title='Average Measurement with Standard Deviation',
            xaxis_title='Time',
            yaxis_title='Average Measurement',
            legend_title='Legend',
            template='plotly_white'
        )

        if log_scale:
            fig.update_yaxes(type="log")

        st.plotly_chart(fig)

# Plot confidence intervals with Plotly
def plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev):
    fig = go.Figure()

    # Plot the 95% confidence interval as a filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([df['Time'], df['Time'][::-1]]),
        y=np.concatenate([lower_bound, upper_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(173, 216, 230, 0.4)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))

    # Plot the observed data
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average'],
        mode='lines',
        name='Observed Data',
        line=dict(color='royalblue')
    ))

    # Plot the standard deviation as a filled area
    fig.add_trace(go.Scatter(
        x=df['Time'].tolist() + df['Time'].tolist()[::-1],
        y=(df['Average'] - std_dev).tolist() + (df['Average'] + std_dev).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(144, 238, 144, 0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Standard Deviation'
    ))

    # Plot the fitted curve
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=y_pred,
        mode='lines',
        name='Fitted Curve',
        line=dict(color='crimson')
    ))

    # Update the layout
    fig.update_layout(
        title='Confidence Intervals with Standard Deviation and Fitted Curve',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )

    return fig  # Instead of st.plotly_chart(fig), return the figure


# Plot selected wells with Plotly
def plot_selected_wells(df, selected_wells):
    fig = go.Figure()
    for well in selected_wells:
        fig.add_trace(go.Scatter(x=df['Time'], y=df[well], mode='lines', name=well))
    fig.update_layout(title="Time vs Selected Well's OD", xaxis_title='Time', yaxis_title='OD', legend_title='Legend', template='plotly_white')
    st.plotly_chart(fig)


# Function to plot Time vs Average with Std Dev
def plot_time_vs_average_with_std(df):
    fig = go.Figure()

    # Add average line
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average'],
        mode='lines',
        name='Average OD'
    ))

    # Add shaded area for Std Dev
    fig.add_trace(go.Scatter(
        x=list(df['Time']) + list(df['Time'])[::-1],
        y=list(df['Average'] + df['Std_Dev']) + list((df['Average'] - df['Std_Dev'])[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Std Dev',
        showlegend=True
    ))

    fig.update_layout(
        title="Time vs Average OD with Std Dev",
        xaxis_title="Time",
        yaxis_title="OD",
        template='plotly_white'
    )

    st.plotly_chart(fig)