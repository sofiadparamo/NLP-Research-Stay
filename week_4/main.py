import numpy as np
from scipy.stats import norm, expon, poisson, gamma
import matplotlib.pyplot as plt

# Example 1: Maximum Likelihood Estimation for Normal Distribution
def mle_normal(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return mu, sigma

data_example1 = [1.2, 2.5, 1.7, 3.3, 4.1]
mu1, sigma1 = mle_normal(data_example1)
print("Example 1 - Normal Distribution:")
print("Estimated mean:", mu1)
print("Estimated standard deviation:", sigma1)
x = np.linspace(-5, 10, 100)
y = norm.pdf(x, mu1, sigma1)
# Draw line at MLE estimate
plt.axvline(mu1, color="red", linestyle="--")
# Draw region for standard deviation
plt.axvspan(mu1 - sigma1, mu1 + sigma1, alpha=0.2, color="red")
plt.plot(x, y)
plt.savefig("assets/normal.png")
plt.close()
print()

# Example 2: Maximum Likelihood Estimation for Exponential Distribution
def mle_exponential(data):
    lam = 1 / np.mean(data)
    return lam

data_example2 = [0.5, 1.3, 2.7, 1.1, 0.8]
lam2 = mle_exponential(data_example2)
print("Example 2 - Exponential Distribution:")
print("Estimated lambda:", lam2)
x = np.linspace(0, 4, 100)
y = expon.pdf(x, scale=1/lam2)
# Draw line at MLE estimate
plt.axvline(lam2, color="red", linestyle="--")
plt.plot(x, y)
plt.savefig("assets/exponential.png")
plt.close()
print()

# Example 3: Maximum Likelihood Estimation for Poisson Distribution
def mle_poisson(data):
    lam = np.mean(data)
    return lam

data_example3 = [2, 4, 3, 5, 2]
lam3 = mle_poisson(data_example3)
print("Example 3 - Poisson Distribution:")
print("Estimated lambda:", lam3)
x = np.arange(0, 10)
y = poisson.pmf(x, lam3)
# Draw line at MLE estimate
plt.axvline(lam3, color="red", linestyle="--")
plt.plot(x, y)
plt.savefig("assets/poisson.png")
plt.close()
print()

# Example 4: Maximum Likelihood Estimation for Gamma Distribution
def mle_gamma(data):
    shape = (np.mean(data) / np.var(data)) ** 2
    scale = np.var(data) / np.mean(data)
    return shape, scale

data_example4 = [5, 10, 15, 20, 25]
shape4, scale4 = mle_gamma(data_example4)
print("Example 4 - Gamma Distribution:")
print("Estimated shape:", shape4)
print("Estimated scale:", scale4)
x = np.linspace(0, 30, 100)
y = gamma.pdf(x, shape4, scale=scale4)
# Draw line at MLE estimate
plt.axvline(shape4 * scale4, color="red", linestyle="--")
plt.plot(x, y)
plt.savefig("assets/gamma.png")
plt.close()
print()

# Example 5: Maximum Likelihood Estimation for Mixture of Normals (2 components)
def mle_mixture_of_normals(data):
    # Assuming equal weights for the two components
    mu1 = np.mean(data)
    sigma1 = np.std(data)
    mu2 = np.mean(data) + np.std(data)
    sigma2 = np.std(data)
    return mu1, sigma1, mu2, sigma2

data_example5 = [1.2, 2.5, 1.7, 3.3, 4.1, 10.2, 9.8, 11.5, 10.9]
mu1_5, sigma1_5, mu2_5, sigma2_5 = mle_mixture_of_normals(data_example5)
print("Example 5 - Mixture of Normals:")
print("Estimated mean (Component 1):", mu1_5)
print("Estimated standard deviation (Component 1):", sigma1_5)
print("Estimated mean (Component 2):", mu2_5)
print("Estimated standard deviation (Component 2):", sigma2_5)
x = np.linspace(-5, 20, 100)
y1 = norm.pdf(x, mu1_5, sigma1_5)
y2 = norm.pdf(x, mu2_5, sigma2_5)
# Draw line at MLE estimate
plt.axvline(mu1_5, color="red", linestyle="--")
plt.axvline(mu2_5, color="green", linestyle="--")
# Draw region for standard deviation
plt.axvspan(mu1_5 - sigma1_5, mu1_5 + sigma1_5, alpha=0.2, color="red")
plt.axvspan(mu2_5 - sigma2_5, mu2_5 + sigma2_5, alpha=0.2, color="green")
plt.plot(x, y1, label="Component 1")
plt.plot(x, y2, label="Component 2")
plt.legend()
plt.savefig("assets/mixture_of_normals.png")
plt.close()