import streamlit as st
import math

def solve_quadratic_equation(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                result = 'The linear equation has infinitely many solutions'
            else:
                result = 'The linear equation has no solution'
        else:
            x = -c / b
            result = 'The linear equation has one solution: %.2f' % x
    else:
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            result = 'The quadratic equation has no real solutions'
        else:
            x1 = (-b + math.sqrt(delta)) / (2 * a)
            x2 = (-b - math.sqrt(delta)) / (2 * a)
            result = 'The quadratic equation has two solutions: x1 = %.2f and x2 = %.2f' % (x1, x2)
    return result

def clear_input():
    st.session_state["quadratic_input"] = {"a": 0.0, "b": 0.0, "c": 0.0}

def solve():
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.25rem;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stButton > button:active {
            background-color: #3e8e41;
        }
        .result {
            margin-top: 1.5rem;
            padding: 1rem;
            background-color: #f5f5f5;
            border-radius: 0.25rem;
            font-size: 1.2rem;
        }
        .result.green {
            background-color: #e6f5e6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Quadratic Equation Solver')
    st.write('Enter coefficients (a, b, c) for the quadratic equation ax^2 + bx + c = 0')

    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        a = st.number_input('a', value=st.session_state.get('quadratic_input', {}).get('a', 0.0))
    with form_col2:
        b = st.number_input('b', value=st.session_state.get('quadratic_input', {}).get('b', 0.0))
    with form_col3:
        c = st.number_input('c', value=st.session_state.get('quadratic_input', {}).get('c', 0.0))

    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        solve_button = st.button('Solve')
    with col2:
        clear_button = st.button('Clear')

    if solve_button:
        result = solve_quadratic_equation(a, b, c)
        st.markdown(f'<div class="result green">{result}</div>', unsafe_allow_html=True)
    elif clear_button:
        clear_input()

    st.session_state["quadratic_input"] = {"a": a, "b": b, "c": c}

