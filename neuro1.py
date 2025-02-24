import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

np.random.seed(42)
weights = np.random.uniform(-0.5, 0.5, size=(2, 2))
b1, b2 = 0.5, 0.7

inputs = np.array([[0.593269992, 0.596884378]])

hidden_layer_input = np.dot(inputs, weights) + np.array([b1, b2])
hidden_layer_output = tanh(hidden_layer_input)

output_weights = np.random.uniform(-0.5, 0.5, size=(2, 2))
output_layer_input = np.dot(hidden_layer_output, output_weights)
output_layer_output = tanh(output_layer_input)

print("Output of the network:")
print(output_layer_output)

root = tk.Tk()
root.title("Neural Network with tanh Activation")

fig, ax = plt.subplots()

ax.text(0.1, 0.5, 'Input Layer\n1X2', fontsize=12, ha='center', bbox=dict(facecolor='yellow', alpha=0.5))
ax.text(0.5, 0.7, f'Hidden Layer\nH1\n{hidden_layer_output[0][0]:.4f}', fontsize=12, ha='center', bbox=dict(facecolor='red', alpha=0.5))
ax.text(0.5, 0.3, f'Hidden Layer\nH2\n{hidden_layer_output[0][1]:.4f}', fontsize=12, ha='center', bbox=dict(facecolor='red', alpha=0.5))

ax.text(0.9, 0.5, f'Output Layer\n{output_layer_output[0][0]:.4f}', fontsize=12, ha='center', bbox=dict(facecolor='blue', alpha=0.5))

ax.annotate("", xy=(0.2, 0.5), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0.2, 0.5), xytext=(0.4, 0.3), arrowprops=dict(arrowstyle="->"))

ax.annotate("", xy=(0.6, 0.7), xytext=(0.8, 0.5), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0.6, 0.3), xytext=(0.8, 0.5), arrowprops=dict(arrowstyle="->"))

ax.axis('off')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

root.mainloop()