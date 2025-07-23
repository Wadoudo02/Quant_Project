from math import exp


def future_descrete_value(x, r, n):
    return x * (1 + r) ** n


def present_descrete_value(x, r, n):
    return x * (1 + r) ** -n


def future_continuous_value(x, r, t):
    return x * exp(r * t)


def present_continuous_value(x, r, t):
    return x * exp(-r * t)


# Value of investment in $

x = 100

# Interest rate 5%

r = 0.05

n = 5  # 5 Years

print("Future value of x (discrete): %s" % future_descrete_value(x, r, n))
print("Present value of x (discrete): %s" % present_descrete_value(x, r, n))
print("Future value of x (continuous): %s" % future_continuous_value(x, r, n))
print("Present value of x (continuous): %s" % present_continuous_value(x, r, n))
