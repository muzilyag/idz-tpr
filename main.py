import numpy as np
import re
from prettytable import PrettyTable

EPS = 1e-9

class LinearProblem:
    def __init__(self, c, f_const, A, b, signs, mode, var_names):
        self.c = np.array(c, dtype=float)
        self.f_const = float(f_const)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signs = signs
        self.mode = mode
        self.var_names = var_names

class SimplexSolver:
    def __init__(self, problem: LinearProblem):
        self.problem = problem
        self.b_vars = []
        self.nb_vars = []
        self.table = None

    def build_initial_algebraic_table(self):
        p = self.problem
        try:
            eq_idx = p.signs.index('=')
        except ValueError:
            eq_idx = 0

        candidates = [j for j in range(len(p.var_names)) if abs(p.A[eq_idx, j]) > EPS]
        if not candidates:
            raise ValueError("В строке '=' все коэффициенты нулевые (или слишком малы)")
        pivot_idx = candidates[0]
        pivot_val = p.A[eq_idx, pivot_idx]
        pivot_var = p.var_names[pivot_idx]

        b_expr = p.b[eq_idx] / pivot_val
        coeffs_expr = {
            p.var_names[j]: p.A[eq_idx, j] / pivot_val
            for j in range(len(p.var_names)) if j != pivot_idx
        }

        self.b_vars = [pivot_var]
        self.nb_vars = [v for v in p.var_names if v != pivot_var]
        rows = [[b_expr] + [coeffs_expr[v] for v in self.nb_vars]]

        max_id = 0
        for v in p.var_names:
            m = re.search(r"\d+", v)
            if m:
                max_id = max(max_id, int(m.group()))
        slack_id = max_id + 1

        for i, sign in enumerate(p.signs):
            if i == eq_idx:
                continue

            a_i_pivot = p.A[i, pivot_idx]
            row_b = p.b[i] - a_i_pivot * b_expr
            row_coeffs = []
            
            for j, v in enumerate(self.nb_vars):
                orig_idx = p.var_names.index(v)
                c_val = p.A[i, orig_idx] - a_i_pivot * coeffs_expr[v]
                row_coeffs.append(c_val)

            if sign == '>=':
                row_b = -row_b
                row_coeffs = [-c for c in row_coeffs]

            s_name = f"x{slack_id}"
            self.b_vars.append(s_name)
            rows.append([row_b] + row_coeffs)
            slack_id += 1

        f_b = p.f_const + (p.c[pivot_idx] * b_expr)
        f_coeffs = []
        for j, v in enumerate(self.nb_vars):
            orig_idx = p.var_names.index(v)
            val = (p.c[pivot_idx] * coeffs_expr[v]) - p.c[orig_idx]
            f_coeffs.append(val)

        rows.append([f_b] + f_coeffs)
        self.table = np.array(rows, dtype=float)

    def is_optimal(self):
        f_coeffs = self.table[-1, 1:]
        if self.problem.mode == 'max':
            return not np.any(f_coeffs < -EPS)
        return not np.any(f_coeffs > EPS)

    def get_infeasible_rows(self):
        return [i for i in range(len(self.b_vars)) if self.table[i, 0] < -EPS]

    def choose_pivot_column(self, infeasible=None):
        if infeasible is None:
            f_coeffs = self.table[-1, 1:]
            if self.problem.mode == 'max':
                return int(np.argmin(f_coeffs)) + 1
            return int(np.argmax(f_coeffs)) + 1

        r_target = infeasible[0]
        cols = [j for j in range(1, self.table.shape[1]) if self.table[r_target, j] < -EPS]
        if not cols:
            return None
        return cols[0]

    def choose_pivot_row(self, col_idx):
        ratios = []
        for i in range(len(self.b_vars)):
            b_val = self.table[i, 0]
            a_val = self.table[i, col_idx]
            if (b_val >= -EPS and a_val > EPS) or (b_val < -EPS and a_val < -EPS):
                ratios.append(b_val / a_val)
            else:
                ratios.append(np.inf)

        row_idx = int(np.argmin(ratios))
        if ratios[row_idx] == np.inf:
            return None, ratios
        return row_idx, ratios

    def solve(self):
        self.build_initial_algebraic_table()
        iteration = 1

        while True:
            print(f"\nИТЕРАЦИЯ {iteration}")
            self.print_table()

            infeasible = self.get_infeasible_rows()
            is_opt = self.is_optimal()

            if is_opt and not infeasible:
                print("\nРЕШЕНИЕ ОПТИМАЛЬНО")
                break

            if not is_opt:
                col_idx = self.choose_pivot_column()
                reason = (
                    f"минимальный отрицательный"
                    if self.problem.mode == 'max'
                    else "максимальный положительный"
                )
                print(f"Выбираем ведущий столбец: {self.nb_vars[col_idx - 1]} "
                      f"({reason} коэффициент в F ({round(self.table[-1, col_idx], 2)}))")
            else:
                r_target = infeasible[0]
                print(f"Решение недопустимо (b < 0 в строке {self.b_vars[r_target]}). Исправляем базис.")
                col_idx = self.choose_pivot_column(infeasible)
                if col_idx is None:
                    print("Задача не имеет допустимых решений")
                    return
                print(f"Выбираем ведущий столбец: {self.nb_vars[col_idx - 1]}")

            print("Расчет отношений b_i / a_ij:")
            row_idx, ratios = self.choose_pivot_row(col_idx)
            for i in range(len(self.b_vars)):
                b_val, a_val = self.table[i, 0], self.table[i, col_idx]
                if ratios[i] != np.inf:
                    print(f"  {self.b_vars[i]}: {round(b_val, 2)} / {round(a_val, 2)} = {round(ratios[i], 2)}")

            if row_idx is None:
                print("Решение не ограничено")
                return

            print(f"Ведущая строка: {self.b_vars[row_idx]}, ведущий элемент: {round(self.table[row_idx, col_idx], 2)}")
            self.jordan_step(row_idx, col_idx)
            iteration += 1

        self.print_result()

    def jordan_step(self, r, c):
        pivot = self.table[r, c]
        new_table = np.zeros_like(self.table)

        new_table[r, c] = 1.0 / pivot
        for j in range(self.table.shape[1]):
            if j != c:
                new_table[r, j] = self.table[r, j] / pivot

        for i in range(self.table.shape[0]):
            if i != r:
                new_table[i, c] = -self.table[i, c] / pivot

        for i in range(self.table.shape[0]):
            if i == r:
                continue
            for j in range(self.table.shape[1]):
                if j == c:
                    continue
                new_table[i, j] = self.table[i, j] - (self.table[i, c] * self.table[r, j] / pivot)

        out_v, in_v = self.b_vars[r], self.nb_vars[c - 1]
        self.b_vars[r], self.nb_vars[c - 1] = in_v, out_v
        self.table = new_table

    def print_table(self):
        pt = PrettyTable()
        pt.field_names = ["Базис", "b"] + self.nb_vars
        for i, bv in enumerate(self.b_vars):
            row = [bv, round(self.table[i, 0], 2)] + [round(val, 2) for val in self.table[i, 1:]]
            pt.add_row(row)
        f_row = ["F", round(self.table[-1, 0], 2)] + [round(v, 2) for v in self.table[-1, 1:]]
        pt.add_row(f_row)
        print(pt)

    def print_result(self):
        print("\nОПТИМАЛЬНЫЙ ПЛАН:")
        final_vals = {v: 0.0 for v in self.problem.var_names}

        for i, bv in enumerate(self.b_vars):
            if bv in final_vals:
                final_vals[bv] = self.table[i, 0]

        real_vars = {}
        for name, val in final_vals.items():
            if "_pos" in name:
                base = name.replace("_pos", "")
                real_vars[base] = real_vars.get(base, 0.0) + val
            elif "_neg" in name:
                base = name.replace("_neg", "")
                real_vars[base] = real_vars.get(base, 0.0) - val
            else:
                real_vars[name] = val

        sorted_keys = sorted(
            real_vars.keys(),
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
        )
        plan_str = ", ".join([f"{k} = {round(real_vars[k], 2)}" for k in sorted_keys])
        print(plan_str)
        print(f"F_opt = {round(self.table[-1, 0], 2)}")


def parse_linear_part(s):
    s = s.replace(" ", "").replace("-", "+-")
    terms = s.split("+")
    coeffs = {}
    const = 0.0

    for t in terms:
        if not t:
            continue
        if 'x' in t:
            parts = t.split('x')
            val_s = parts[0]
            if val_s == "" or val_s == "+":
                val = 1.0
            elif val_s == "-":
                val = -1.0
            else:
                val = float(val_s)

            idx_match = re.search(r"\d+", parts[1])
            if idx_match:
                idx = int(idx_match.group())
                coeffs[idx] = coeffs.get(idx, 0.0) + val
        else:
            try:
                const += float(t)
            except:
                pass

    return coeffs, const


def get_input_parsed():
    print("Введите задачу. Пример:\n"
          "F = 1x1 + 4x2 + 1x3 -> max\n"
          "-1x1 + 2x2 + 1x3 = 4\n"
          "3x1 + 1x2 + 2x3 <= 9\n"
          "2x1 + 3x2 + 1x3 >= 6\n"
          "x1 >= 0\n"
          "x2 >= 0\nВведите 'solve' для начала.\n")

    f_line = input("F = ")
    f_match = re.split(r'->|=', f_line)
    mode = f_match[-1].strip().lower()
    expr_to_parse = f_match[1] if len(f_match) > 2 else f_match[0].split('=')[-1]
    f_coeffs_dict, f_const = parse_linear_part(expr_to_parse)

    constraints_raw = []
    while True:
        line = input("Ограничение: ").strip()
        if line.lower() == 'solve':
            break
        constraints_raw.append(line)

    non_neg_indices = set()
    structural_constraints = []
    all_indices = set(f_coeffs_dict.keys())

    for c_line in constraints_raw:
        nn_match = re.match(r'x(\d+)\s*>=\s*0', c_line)
        if nn_match:
            non_neg_indices.add(int(nn_match.group(1)))
            continue

        for s_op in ['<=', '>=', '=']:
            if s_op in c_line:
                left, right = c_line.split(s_op)
                c_coeffs, c_const = parse_linear_part(left)
                all_indices.update(c_coeffs.keys())
                structural_constraints.append((c_coeffs, s_op, float(right) - c_const))
                break

    num_vars = max(all_indices) if all_indices else 0
    non_neg_indices.update(range(1, num_vars + 1))

    print("\nИСХОДНАЯ ЗАДАЧА")
    print(f"Целевая функция: F({mode}) = {f_line.split('->')[0].replace('F=', '').strip()}")
    print("Ограничения:")
    for coeffs, op, val in structural_constraints:
        parts = [f"{v}x{i}" for i, v in coeffs.items()]
        print(f"  {' + '.join(parts)} {op} {val}")
    print("Условия неотрицательности:")
    for i in range(1, num_vars + 1):
        cond = ">= 0" if i in non_neg_indices else "свободна (может быть < 0)"
        print(f"  x{i} {cond}")

    var_map = {}
    new_var_names = []
    curr_idx = 0

    for i in range(1, num_vars + 1):
        var_map[i] = [curr_idx]
        new_var_names.append(f"x{i}")
        curr_idx += 1 

    new_num_vars = len(new_var_names)
    new_c = [0.0] * new_num_vars
    for i, val in f_coeffs_dict.items():
        indices = var_map[i]
        new_c[indices[0]] = val
        if len(indices) > 1:
            new_c[indices[1]] = -val

    new_A, new_b, new_signs = [], [], []
    for coeffs, op, val in structural_constraints:
        row = [0.0] * new_num_vars
        for i, c_val in coeffs.items():
            indices = var_map[i]
            row[indices[0]] = c_val
            if len(indices) > 1:
                row[indices[1]] = -c_val
        new_A.append(row)
        new_b.append(val)
        new_signs.append(op)

    problem = LinearProblem(new_c, f_const, new_A, new_b, new_signs, mode, new_var_names)
    return problem


if __name__ == "__main__":
    try:
        problem = get_input_parsed()
        solver = SimplexSolver(problem)
        solver.solve()
    except Exception as e:
        print(f"Ошибка: {e}")