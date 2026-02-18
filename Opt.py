"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import numpy as np
import sympy as sp
#from main import gradiente
from try2 import gradiente

from rxconfig import config


class State(rx.State):
    """The app state."""
    
    # Input fields
    function_input: str = "(x-4)**2 + 3*x*y + 5*y**2"
    variables_input: str = "x, y"
    x0_input: str = "1.0, 2.0"
    #alpha_input: str = "0.01"
    tol_input: str = "1"
    max_iter_input: str = "1000"
    modo: str = "min"
    
    # Results
    solucion: str = ""
    iteraciones: str = ""
    error_message: str = ""
    resultados = []

    def set_modo(self, value: str):
        self.modo = value

    def calculate(self):
        """Calculate the gradient descent result."""
        try:
            # Clear previous results/errors
            self.error_message = ""
            self.solucion = ""
            self.iteraciones = ""
            self.resultados = []

            # Parse variables
            var_strings = [v.strip() for v in self.variables_input.split(',')]
            if not var_strings or var_strings == ['']:
                raise ValueError("Debe ingresar al menos una variable.")
                
            variables = sp.symbols(var_strings)
            
            # Parse function
            f = sp.sympify(self.function_input)
            
            # Parse x0
            x0_values = [float(x.strip()) for x in self.x0_input.split(',')]
            x0 = np.array(x0_values)
            
            if len(x0) != len(variables):
                raise ValueError(f"El número de valores iniciales ({len(x0)}) no coincide con el número de variables ({len(variables)}).")

            # Parse numeric params
            try:
                #alpha = float(self.alpha_input)
                tol = float(self.tol_input)
                max_iter = int(self.max_iter_input)
            except ValueError:
                raise ValueError("Alpha, Tolerancia y Iteraciones deben ser números válidos.")

            # Call gradiente function
            sol = gradiente(
                f,
                variables,
                x0,
                tol,
                max_iter,
                self.modo
            )
            
            self.solucion = str(sol)
            # self.iteraciones = str(it)
            # self.resultados = str(res)

        except Exception as e:
            self.error_message = str(e)

def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.center(
        rx.vstack(
            rx.heading("Ascenso y Descenso Acelerado", font_size="2em", padding="1em"),
            
            rx.card(
                rx.vstack(
                    rx.text("Función f(x, y, ...):"),
                    rx.input(
                        placeholder="(x-4)**2 + 3*x*y + 5*y**2",
                        value=State.function_input,
                        on_change=State.set_function_input,
                        width="100%"
                    ),
                    
                    rx.text("Variables (separadas por coma):"),
                    rx.input(
                        placeholder="x, y",
                        value=State.variables_input,
                        on_change=State.set_variables_input,
                        width="100%"
                    ),
                    
                    rx.text("Punto inicial x0 (separado por coma):"),
                    rx.input(
                        placeholder="1.0, 2.0",
                        value=State.x0_input,
                        on_change=State.set_x0_input,
                        width="100%"
                    ),
                    
                    rx.hstack(
                        # rx.vstack(
                        #     rx.text("Alpha (tasa de aprendizaje):"),
                        #     rx.input(
                        #         value=State.alpha_input,
                        #         on_change=State.set_alpha_input,
                        #     ),
                        # ),
                        rx.vstack(
                            rx.text("Tolerancia:"),
                            rx.input(
                                value=State.tol_input,
                                on_change=State.set_tol_input,
                            ),
                        ),
                        width="100%",
                        justify="between"
                    ),
                    
                    rx.hstack(
                        rx.vstack(
                            rx.text("Máx. Iteraciones:"),
                            rx.input(
                                value=State.max_iter_input,
                                on_change=State.set_max_iter_input,
                            ),
                        ),
                        rx.vstack(
                            rx.text("Modo:"),
                            rx.select(
                                ["min", "max"],
                                value=State.modo,
                                on_change=State.set_modo,
                            ),
                        ),
                        width="100%",
                        justify="between"
                    ),

                    rx.button("Calcular", on_click=State.calculate, width="100%", color_scheme="blue", margin_top="1em"),
                ),
                width="50vw",
                max_width="600px",
                padding="2em",
            ),
            
            rx.cond(
                State.error_message != "",
                rx.callout(
                    State.error_message,
                    icon="alert_triangle",
                    color_scheme="red",
                    width="50vw",
                    max_width="600px",
                )
            ),
            rx.cond(
                State.solucion != "",
                rx.card(
                    rx.vstack(
                        rx.heading("Resultados", size="4"),
                        rx.text(f"{State.solucion}", white_space="pre-wrap"),
                        # rx.text(
                        #     f"{State.resultados}", 
                        #     white_space="pre-wrap"
                            
                        # ),
                    ),
                    width="50vw",
                    max_width="600px",
                    border_color="green",
                )
            ),
            
            rx.box(
                rx.text("Alexandra Hidalgo y Fernando Figuera", font_size="0.9em", color="gray"),
                padding="1em"
            ),
            
            align_items="center",
            
        ),
        height="auto",
        #background_color="#f5f5f5",
    )


app = rx.App()
app.add_page(index)
