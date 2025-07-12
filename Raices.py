import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify, diff
import base64
import pandas as pd
import sympy as sp


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("logo_tecnm.png")

st.markdown(f"""
    <div style='text-align: center; margin-top: 10px; margin-bottom: -20px;'>
        <img src="data:image/png;base64,{img_base64}" width="250">
    </div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Raices de Funciones", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>√ Raices de Funciones √</h1>
    <h3 style='text-align: center; color: gray;'>Tecnológico Nacional de México</h3>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px; margin-top: 20px;'>"
    "Desarrollado en el Laboratorio de Simulación Matemática del Instituto Tecnológico de Morelia por el Ing. Juan David López Regalado"
    "</p>",
    unsafe_allow_html=True
)

# Entrada del usuario
method = st.selectbox("Selecciona el método:", ["Bisección",  "Newton-Raphson", "Secante", "Regla Falsa"])

with st.expander("Ver teoría del método seleccionado..."):
    if method == "Bisección":
        st.markdown(r"""
        ### 🔴 Método de Bisección:
        Se parte de un intervalo $[a, b]$ donde la función cambia de signo:
        $$
        f(a) \cdot f(b) < 0
        $$
        Se calcula el punto medio:
        $$
        c = \frac{a + b}{2}
        $$
        Luego se evalúa $f(c)$ y se reemplaza el intervalo por $[a, c]$ o $[c, b]$ según el signo de $f(c)$. Se repite hasta que el intervalo sea suficientemente pequeño. Es seguro, pero puede ser lento.
        """)

    elif method == "Newton-Raphson":
        st.markdown(r"""
        ### 🟡 Método de Newton-Raphson
        Usa la derivada para mejorar la aproximación:
        $$
        x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
        $$
        Este método es muy rápido si la estimación inicial está cerca de la raíz, pero requiere calcular la derivada $f'(x)$.
        """)

    elif method == "Secante":
        st.markdown(r"""
        ### 🟠 Método de la Secante
        No usa la derivada explícitamente, sino una pendiente aproximada entre dos puntos:
        $$
        x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}
        $$
        Requiere dos valores iniciales, y es útil cuando no se tiene o no se quiere calcular $f'(x)$.    
        """)

    elif method == "Regla Falsa":
        st.markdown(r"""
        ### 🟤 Método de la Regla Falsa (Falsa Posición)
        Como el método de bisección, se parte de un intervalo $[a, b]$ con cambio de signo. En vez de usar el punto medio, se usa la intersección de la recta que une $(a, f(a))$ y $(b, f(b))$ con el eje $x$:
        $$
        c = b - f(b) \cdot \frac{b - a}{f(b) - f(a)}
        $$
        Luego se reemplaza el intervalo según el signo de $f(c)$. Suelen converger más rápido que la bisección, manteniendo la seguridad.
        """)

func_str = st.text_input("Escribe la función f(x):", value="x**3+x-2")

x = symbols('x')

try:
    func_expr = sympify(func_str)
    f = lambdify(x, func_expr, 'numpy')
    df_expr = diff(func_expr, x)
    df = lambdify(x, df_expr, 'numpy')

    st.latex(f"f(x) = {func_expr}")


    if method == "Bisección":
    
        st.subheader("⚙️ Parámetros del método")

        # Parámetros comunes
        a = st.number_input("Valor inicial a:", value=1.0)
        b = st.number_input("Valor inicial b:", value=2.0)
        tol = st.number_input("Tolerancia:", value=1e-3, format="%e")
        max_iter = st.slider("Iteraciones máximas:", min_value=1, max_value=100, value=20)

        iter_data = []
        raiz_aprox = None
        errores = []
        iter_x_vals = []
        iter_fx_vals = []
        raiz_aprox = None
        iter_encontrada = None
        
        tol = tol 
        fa, fb = f(a), f(b)
        if f(a) * f(b) >= 0:
            st.warning("La multiplicación f(a)*f(b) debe ser menor a 0")
        else:
            iter_data = []
            iter_x_vals = []
            iter_fx_vals = []
            errores = []
            
            for i in range(1, max_iter + 1):
                c = (a + b) / 2
                fc = f(c)
                error = abs(b - a) / 2
                
                iter_data.append((i, a, b, c, fc, error))
                iter_x_vals.append(c)
                iter_fx_vals.append(fc)
                errores.append(error)
                
                raiz_aprox = c
                iter_encontrada = i
                
                if error < tol or abs(fc) < tol:
                    break  # aquí se detiene si se cumple tolerancia
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            
            df_iter = pd.DataFrame(iter_data, columns=["Iteración", "a (x_n)", "b (x_{n+1})", "c", "f(c)", "Error"])

            st.markdown("### 📊 Tabla de iteraciones")
            st.dataframe(df_iter.style.format({
                "a (x_n)": "{:.6f}",
                "b (x_{n+1})": "{:.6f}",
                "c": "{:.6f}",
                "f(c)": "{:.4e}",
                "Error": "{:.6e}"
            }).set_table_styles([
                {'selector': 'th, td', 'props': [('font-size', '10px'), ('padding', '4px 8px')]}
            ]))


        
    elif method == "Newton-Raphson":
        st.subheader("⚙️ Parámetros del método")

        # Parámetros comunes
        x0 = st.number_input("Valor inicial x₀", value=1.0)
        tol = st.number_input("Tolerancia:", value=1e-3, format="%e")
        max_iter = st.slider("Iteraciones máximas:", min_value=1, max_value=100, value=20)
        st.markdown("""
        > ⚠️ **Importante:** Aunque el método de Newton-Raphson solo utiliza el valor inicial `x₀` para comenzar el proceso, los valores `X+` y `X−` aquí definidos se utilizan **únicamente** para establecer la escala del gráfico de la función.  
        > No influyen en el cálculo de la raíz.
        """)
        st.subheader("Escala del gráfico")

        a = st.number_input("Límite inferior para graficar (X−)", value=1.0)+1
        b = st.number_input("Límite superior para graficar (X+)", value=2.0)-1

        iter_data = []
        raiz_aprox = None
        errores = []
        iter_x_vals = []
        iter_fx_vals = []
        raiz_aprox = None
        iter_encontrada = None
        
        for i in range(1, max_iter + 1):
            fx = f(x0)
            dfx = df(x0)
            if dfx == 0:
                st.warning("Derivada cero. No se puede continuar.")
                break
            x1 = x0 - fx / dfx
            error = abs(x1 - x0)
            iter_data.append((i, x0, x1, fx, error))
            iter_x_vals.append(x0)
            iter_fx_vals.append(fx)
            errores.append(error)
            if error < tol:
                raiz_aprox = x1
                iter_encontrada = i 
                break
            x0 = x1
        # Crear dataframe con las columnas que pides
        df_iter = pd.DataFrame(iter_data, columns=["Iteración", "a (x_n)", "b (x_{n+1})", "f(x_n)", "Error"])

        # Mostrar tabla en Streamlit con formato
        st.markdown("### 📊 Tabla de iteraciones")
        st.dataframe(df_iter.style.format({
            "a (x_n)": "{:.6f}",
            "b (x_{n+1})": "{:.6f}",
            "f(x_n)": "{:.4e}",
            "Error": "{:.6e}"
        }))

    elif method == "Secante":
        st.subheader("⚙️ Parámetros del método")

        # Parámetros de entrada
        a = st.number_input("Valor inicial a (x₀):", value=1.0)
        b = st.number_input("Valor inicial b (x₁):", value=2.0)
        tol = st.number_input("Tolerancia:", value=1e-3, format="%e")
        max_iter = st.slider("Iteraciones máximas:", min_value=1, max_value=100, value=20)

        # Preparar iteraciones
        iter_data = []
        errores = []
        iter_x_vals = []
        iter_fx_vals = []
        raiz_aprox = None
        iter_encontrada = None

        x0, x1 = a, b

        for i in range(1, max_iter + 1):
            fx0, fx1 = f(x0), f(x1)

            if fx1 - fx0 == 0:
                st.warning(f"⚠️ División por cero en la iteración {i}. No se puede continuar.")
                break

            try:
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                error = abs(x2 - x1)

                iter_data.append((i, x0, x1, fx1, x2, error))
                iter_x_vals.append(x2)
                iter_fx_vals.append(f(x2))
                errores.append(error)

                if error < tol:
                    raiz_aprox = x2
                    iter_encontrada = i
                    break

                x0, x1 = x1, x2
            except Exception as e:
                st.error(f"❌ Error en la iteración {i}: {e}")
                break

        # Crear y mostrar la tabla de iteraciones
        if iter_data:
            df_iter = pd.DataFrame(iter_data, columns=[
                "Iteración", "x₀", "x₁", "f(x₁)", "x₂ (nuevo)", "Error"
            ])
            st.markdown("### 📊 Tabla de iteraciones")
            st.dataframe(df_iter.style.format({
                "x₀": "{:.6f}",
                "x₁": "{:.6f}",
                "f(x₁)": "{:.4e}",
                "x₂ (nuevo)": "{:.6f}",
                "Error": "{:.6e}"
            }))



    elif method == "Regla Falsa":
        
        st.subheader("⚙️ Parámetros del método")

        # Parámetros comunes
        a = st.number_input("Valor inicial a:", value=1.0)
        b = st.number_input("Valor inicial b:", value=2.0)
        tol = st.number_input("Tolerancia:", value=1e-3, format="%e")
        max_iter = st.slider("Iteraciones máximas:", min_value=1, max_value=100, value=20)

        iter_data = []
        raiz_aprox = None
        errores = []
        iter_x_vals = []
        iter_fx_vals = []
        raiz_aprox = None
        iter_encontrada = None

        fa, fb = f(a), f(b)
        if fa * fb > 0:
            st.warning("⚠️ f(a) y f(b) deben tener signos opuestos para garantizar una raíz en el intervalo.")
        else:
            for i in range(1, max_iter + 1):
                c = b - fb * (b - a) / (fb - fa)
                fc = f(c)
                error = abs(fc)
                iter_data.append((i, a, b, c, fc, error))
                iter_x_vals.append(c)
                iter_fx_vals.append(fc)
                errores.append(error)

                if error < tol:
                    raiz_aprox = c
                    iter_encontrada = i 
                    break

                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc

            # Crear dataframe
            df_iter = pd.DataFrame(iter_data, columns=["Iteración", "a", "b", "c (x)", "f(c)", "Error"])

            # Mostrar tabla con formato bonito
            st.markdown("### 📊 Tabla de iteraciones")
            st.dataframe(df_iter.style.format({
                "a": "{:.6f}",
                "b": "{:.6f}",
                "c (x)": "{:.6f}",
                "f(c)": "{:.4e}",
                "Error": "{:.6e}"
            }))

    # Mostrar resultados
    st.subheader("📌 Resultados")
    if raiz_aprox is not None:
        st.success(f"Raíz aproximada: {raiz_aprox:.6f} en la iteración {iter_encontrada}")
    else:
        st.info("No se encontró raíz en el número máximo de iteraciones.")

    st.subheader("📈 Gráfica de f(x)")

    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = np.linspace(a - 1, b + 1, 500)
    y_vals = f(x_vals)
    ax.plot(x_vals, y_vals, label='f(x)', color='blue')
    
    if raiz_aprox is not None:
        ax.axvline(raiz_aprox, color='red', linestyle='--', label='Raíz') 
        ax.plot(raiz_aprox, 0, 'ro', label='Punto raíz')
        ax.text(raiz_aprox, 0, f'{raiz_aprox:.6f}', color='black', fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right')


    ax.axhline(0, color='black', linestyle='--')

    ax.set_title("f(x)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    n_iters = len(iter_x_vals)
    cols = 2
    rows = (n_iters + 1) // cols 

    fig2, axs = plt.subplots(rows, cols, figsize=(10, 4*rows), squeeze=False)

    x_vals = np.linspace(a - 1, b + 1, 500)
    y_vals = f(x_vals)

    for i, ax in enumerate(axs.flat):
        if i < n_iters:
            x_approx = iter_x_vals[i]
            y_approx = iter_fx_vals[i]

            ax.plot(x_vals, y_vals, color='blue')
            ax.axvline(x_approx, color='red', linestyle='--')
            ax.plot(x_approx, y_approx, 'ro')
            ax.axhline(0, color='black', linestyle='--')
            ax.text(x_approx, 0, f'{x_approx:.6f}', color='black', fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right')
            ax.set_title(f"Iteración {i+1}")
            ax.grid(True)
        else:
            ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")
