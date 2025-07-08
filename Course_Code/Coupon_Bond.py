import math

class CouponBond:
    def __init__(self, principle, rate, maturity, interest_rate):

        # Principle amount
        self.principle = principle

        self.rate = rate 
        # date to maturity
        self.maturity = maturity
        # market interest rate (discounting)
        self.interest_rate = interest_rate

    def present_value(self, x, n):
        #return x / (1 + self.interest_rate)**n # Discrete
        return x * math.exp(-self.interest_rate * n) # Continuous
    
    def calculate_price(self):

        price = 0

        # discount the coupon payments

        for t in range(1, self.maturity+1):
            price += self.present_value(self.principle * self.rate, t)
            #print(price)

        # discount principle amoint

        price += self.present_value(self.principle, self.maturity)

        return price
bond = CouponBond(1000,0.1,3,0.04)    
print("Price of the bond: %.2f" % bond.calculate_price())