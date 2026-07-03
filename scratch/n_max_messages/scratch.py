# %%
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np

# Generate mock data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create an interactive plot
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()


# %%
