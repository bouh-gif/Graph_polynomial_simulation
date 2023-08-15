
# importing the required module
# importing SymPy Library
import sympy
import numpy
import matplotlib.pyplot as plt
# print(numpy.linspace(2, 3, num=5))
# creating class square

class A(sympy.Function):
  @classmethod
  # defining subclass function with 'eval()' class method
  def eval(cls, p, m, n, coefs):
    result = []
    for element in p:
        # Core of the function : the algebraic expression
        # m=1 # size of any minimum cut-set (see collection M)
        # n=3 # number of edges
        # #values of coefficients A_i, given by the number of cut-sets of size i, with i between m and n included
        # coefs = [1, 3, 1]
        # changement de variable : x = p/(1-p) et p = element
        x = element/(1.0-element)

        index = list(range(m, n+1))
        sum=0
        for  i in index:
            # Evaluate sum = A(x) = A(p/(1-p)) :
            A_i = coefs[i-1]
            # sum = sum + A_i * pow(x, i)
            sum = sum + A_i*(pow(element, i))*(pow(1-element, n-i))
        # Evaluate O(p)
        # result = pow((1.0-p), n) * sum

        result.append(sum)
    return result
  @classmethod
  # defining subclass function with 'eval()' class method
  def plot(cls, x, y):
    ### PLOT
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('p')
    # naming the y axis
    plt.ylabel('O(p)')

    # giving a title to my graph
    plt.title('Reliability polynomial O(p)')

    # Add gridlines to the plot
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    # grid(b=True, which='major', color='b', linestyle='-')

    # Save the file and show the figure
    plt.savefig("plotted_polynomial.png")

## Call of the function
# x = []
# # Get p/(1-p)
# for element in p:
#    x.append(element/(1.0-element)) # x = p/(1-p)


m=1 # size of any minimum cut-set (see collection M)
n=3 # number of edges
# values of coefficients A_i, given by the number of cut-sets of size i, with i between m and n included
coefs = [1, 3, 1]

p = numpy.linspace(0.01, 0.99, num=1000)
O_p = A.eval(p, m, n, coefs)
A.plot(p, O_p)
print("Here is p : ",p)

print("Here is O(p) : ", O_p)

# ### PLOT
# # convert y-axis to Logarithmic scale
# plt.yscale("log")
# # plotting the points
# plt.plot(p, O_p)

# # naming the x axis
# plt.xlabel('p')
# # naming the y axis
# plt.ylabel('O(p)')

# # giving a title to my graph
# plt.title('Reliability polynomial O(p)')

# # Add gridlines to the plot
# plt.grid(b=True, which='major', linestyle='-')
# plt.grid(b=True, which='minor', linestyle='--')
# # grid(b=True, which='major', color='b', linestyle='-')

# # Save the file and show the figure
# plt.savefig("plotted_polynomial.png")

# # function to show the plot
# plt.show()
