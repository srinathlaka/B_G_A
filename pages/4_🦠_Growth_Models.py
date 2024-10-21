import streamlit as st

def add_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f9f9f9;
            padding: 20px;
        }
        .stApp {
            font-family: "Arial", sans-serif;
            background-color: #f5f5f5;
        }
        .title {
            color: #4CAF50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #2E86C1;
        }
        .equation {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

def app():
    add_custom_css()

    st.markdown("<h1 class='title'>Bacterial Growth Models</h1>", unsafe_allow_html=True)

    # Exponential Growth Model Section
    st.write("## Exponential Growth Model")
    st.write("The **exponential growth model** describes a scenario where the population grows at a constant rate over time.")
    
    with st.expander("Show Equations"):
        st.markdown("<div class='equation'>", unsafe_allow_html=True)
        st.latex(r'\frac{dX}{dt} = \mu X')
        st.latex(r'X(t) = X_0 e^{\mu t}')
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("""
    - **$X_0$**: Initial bacterial biomass
    - **$\mu$**: Growth rate
    - The population grows exponentially as time $t$ increases.
    """)

    # Logistic Growth Model Section
    st.write("## Logistic Growth Model")
    st.write("The **logistic growth model** includes a saturation point, where the population growth slows down as it reaches the carrying capacity.")

    with st.expander("Show Equations"):
        st.markdown("<div class='equation'>", unsafe_allow_html=True)
        st.latex(r'\frac{dX}{dt} = \mu X \left(1 - \frac{X}{K}\right)')
        st.latex(r'X(t) = \frac{X_0 e^{\mu t}}{1 + \frac{X_0}{K} \left(e^{\mu t} - 1\right)}')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("""
    - **$X_0$**: Initial biomass
    - **$\mu$**: Growth rate
    - **$K$**: Saturation constant (carrying capacity)
    - The logistic model shows that growth slows down as it approaches $K$, representing resource limits.
    """)

    # Baranyi Model Section
    st.write("## Baranyi Growth Model")
    st.write("The **Baranyi model** adds a lag phase where the bacteria adapt to new conditions before exponential growth.")

    with st.expander("Show Equations"):
        st.markdown("<div class='equation'>", unsafe_allow_html=True)
        st.latex(r'\frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X')
        st.latex(r'\frac{dq}{dt} = \mu q')
        st.latex(r'X(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0}')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("""
    - **$X_0$**: Initial biomass
    - **$\mu$**: Growth rate
    - **$q_0$**: Initial physiological state
    - This model captures the lag phase before exponential growth begins.
    """)

    # Lag-Exponential-Saturation Model Section
    st.write("## Lag-Exponential-Saturation Growth Model")
    st.write("This model combines the lag, exponential, and saturation phases in bacterial growth.")

    with st.expander("Show Equations"):
        st.markdown("<div class='equation'>", unsafe_allow_html=True)
        st.latex(r'\frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X \left(1 - \frac{X}{K}\right)')
        st.latex(r'\frac{dq}{dt} = \mu q')
        st.latex(r'X(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0 - q_0 \frac{X_0}{K} + \frac{q_0 X_0}{K} e^{\mu t}}')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("""
    - **$X_0$**: Initial biomass
    - **$q_0$**: Physiological state
    - **$\mu$**: Growth rate
    - **$K$**: Carrying capacity
    - This model represents a more detailed understanding of bacterial growth by including all phases.
    """)

if __name__ == "__main__":
    app()
