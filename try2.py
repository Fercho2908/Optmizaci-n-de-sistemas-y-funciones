import numpy as np
import sympy as sp

def gradiente(f, variables, x0, tol=1, max_iter=50, modo="min"):
    # 1. Gradiente simbólico (una sola vez)
    grad_f = [sp.diff(f, v) for v in variables]

    alpha = sp.symbols('alpha', real=True)

    xk = np.array(x0, dtype=float)
    points = ""

    for k in range(max_iter):
        
        # 2. Evaluar gradiente en x_k
        grad_eval = np.array(
            [g.subs(dict(zip(variables, xk))) for g in grad_f],
            dtype=float
        )

        # Criterio de parada
        if np.linalg.norm(grad_eval) < tol:
            print(f"Convergió en {k} iteraciones")
            print(points)
            
            return points
        
        points += f"Iteración {k+1}: {xk}\n"

        # 3. Construir x_{k+1} = x_k +/- alpha * grad
        if modo == "min":
            x_next = [
                xk[i] - alpha * grad_eval[i]
                for i in range(len(xk))
            ]
        elif modo == "max":
            x_next = [
                xk[i] + alpha * grad_eval[i]
                for i in range(len(xk))
            ]
        else:
            raise ValueError("modo debe ser 'min' o 'max'")

        # 4. Sustituir x_{k+1} en la función original
        sustitucion = dict(zip(variables, x_next))
        f_alpha = f.subs(sustitucion)

        # 5. Derivar respecto a alpha
        df_dalpha = sp.diff(f_alpha, alpha)

        # 6. Resolver df/dalpha = 0
        soluciones = sp.solve(df_dalpha, alpha)

        if not soluciones:
            raise ValueError("No se pudo despejar alpha")

        alpha_k = float(soluciones[0])

        # 7. Sustituir alpha en x_{k+1}
        xk = np.array(
            [xi.subs(alpha, alpha_k) for xi in x_next],
            dtype=float
        )

    print("Máximo de iteraciones alcanzado")
    return xk

# x, y = sp.symbols('x y')
# f = (x-4)**2 + 3*x*y + 5*y**2

# x0 = np.array([1.0, 2.0])

# gradiente(f, [x, y], x0)

