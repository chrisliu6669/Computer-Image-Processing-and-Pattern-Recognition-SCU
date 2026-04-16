import sympy as sp


x = sp.symbols('x')

# ===================== Logistic 函数 (Sigmoid) =====================
def func_logistic(x_val):
    return 1 / (1 + sp.exp(-x_val))


f_log = func_logistic(x)
f_log_deriv = sp.diff(f_log, x)
f_log_val = f_log_deriv.evalf(subs={x: 1.0})

# ===================== Tanh 函数 =====================
def func_tanh(x_val):
    return (sp.exp(x_val) - sp.exp(-x_val)) / (sp.exp(x_val) + sp.exp(-x_val))


f_tanh = func_tanh(x)
f_tanh_deriv = sp.diff(f_tanh, x)
f_tanh_val = f_tanh_deriv.evalf(subs={x: 1.0}) # 在 x=1.0 处求值

# ===================== ReLU 函数 =====================
def func_relu(x_val):
    return sp.Max(0, x_val)


f_relu = func_relu(x)
f_relu_deriv = sp.diff(f_relu, x)
f_relu_val = f_relu_deriv.evalf(subs={x: 1.0})

# ===================== ELU 函数 =====================
def func_elu(x_val, gamma=1):
    positive_part = x_val
    negative_part = gamma * (sp.exp(x_val) - 1)
    return sp.Piecewise((positive_part, x_val > 0), (negative_part, True))


f_elu = func_elu(x)
f_elu_deriv = sp.diff(f_elu, x)
f_elu_val = f_elu_deriv.evalf(subs={x: 1.0}) # 在 x=1.0 处求值

# ===================== SoftPlus 函数 =====================
def func_softplus(x_val):
    return sp.log(1 + sp.exp(x_val))

# 求导与计算
f_softplus = func_softplus(x)
f_softplus_deriv = sp.diff(f_softplus, x)
f_softplus_val = f_softplus_deriv.evalf(subs={x: 1.0}) # 在 x=1.0 处求值

# ===================== 输出结果 =====================
print("=" * 50)
print(f"[Logistic] 导数符号: {f_log_deriv}")
print(f"[Logistic] x=1.0 时导数值: {f_log_val:.6f}")

print("\n" + "=" * 50)
print(f"[Tanh] 导数符号: {f_tanh_deriv}")
print(f"[Tanh] x=1.0 时导数值: {f_tanh_val:.6f}")

print("\n" + "=" * 50)
print(f"[ReLU] 导数符号: {f_relu_deriv}")
print(f"[ReLU] x=1.0 时导数值: {f_relu_val:.6f}")

print("\n" + "=" * 50)
print(f"[ELU] 导数符号: {f_elu_deriv}")
print(f"[ELU] x=1.0 时导数值: {f_elu_val:.6f}")

print("\n" + "=" * 50)
print(f"[SoftPlus] 导数符号: {f_softplus_deriv}")
print(f"[SoftPlus] x=1.0 时导数值: {f_softplus_val:.6f}")
print("=" * 50)