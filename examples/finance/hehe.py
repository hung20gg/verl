import math

def chi2_pdf(x, df=9):
    if x < 0:
        return 0.0
    
    coef = 1.0 / ( (2 ** (df / 2.0)) * math.gamma(df / 2.0) )
    return coef * (x ** (df / 2.0 - 1.0)) * math.exp(-x / 2.0)


# ===== Chạy =====
x = float(input("Nhap x: "))
print("f(x) =", chi2_pdf(x, df=9))