
# # importing SymPy Library
import sympy
import numpy
# print(numpy.linspace(2, 3, num=5))
# creating class square
coefs = [1, 3, 1]
class A(sympy.Function):
  @classmethod
  # defining subclass function with 'eval()' class method
  def eval(cls, x):
    result = []
    for element in x:
        # Core of the function : the algebraic expression
        m=1
        n=3
        index = list(range(m, n+1))
        sum=0
        for  i in index: 
            sum = sum + coefs[i-1]*(element**i)*((1-element)**(n-i))
        result.append(sum)
        # result.append(element**2)      
    return result
## Call of the function
p = numpy.linspace(0.01, 0.99, num=10)
x = []
# Get p/(1-p)
for element in p:
   x.append(element/(1.0-element)) # p/(1-p)

result = A.eval(p)
print("Here is p : ",p)

print("Here is result : ",result)
