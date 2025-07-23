class ZeroCouponBonds:
    def __init__(self, principle, maturity, interest_rate):
        # Principle amount
        self.principle = principle
        # date to maturity
        self.maturity = maturity
        # market interest rate (discounting)
        self.interest_rate = interest_rate

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n

    def calculate_price(self):
        return self.present_value(self.principle, self.maturity)


bond = ZeroCouponBonds(1000, 2, 0.04)
print("Price of the bond: %.2f" % bond.calculate_price())
