#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Given data
P_w1 = 0.6  # Prior probability of class ω1
P_w2 = 0.4  # Prior probability of class ω2

mu1, sigma1 = 5, 1.5  # Mean and standard deviation for class ω1
mu2, sigma2 = 8, 2.0  # Mean and standard deviation for class ω2

# Define the range of x values
x_values = np.linspace(0, 15, 100)

# Compute the class-conditional probabilities p(x|ω1) and p(x|ω2)
p_x_given_w1 = stats.norm.pdf(x_values, mu1, sigma1)
p_x_given_w2 = stats.norm.pdf(x_values, mu2, sigma2)

# Compute the total probability P(x)
P_x = (p_x_given_w1 * P_w1) + (p_x_given_w2 * P_w2)

# Compute the posterior probabilities using Bayes' theorem
P_w1_given_x = (p_x_given_w1 * P_w1) / P_x
P_w2_given_x = (p_x_given_w2 * P_w2) / P_x

# Plot the posterior probabilities
plt.figure(figsize=(8, 6))
plt.plot(x_values, P_w1_given_x, label=r'$P(\omega_1 | x)$', color='blue')
plt.plot(x_values, P_w2_given_x, label=r'$P(\omega_2 | x)$', color='red')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.title('Posterior Probabilities $P(\omega_1 | x)$ and $P(\omega_2 | x)$')
plt.legend()
plt.grid(True)
plt.show()

# Return computed values
P_w1_given_x, P_w2_given_x


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Question 1: Bayes Classifier
# Given data
P_w1 = 0.6
P_w2 = 0.4
mu1, sigma1 = 5, 1.5
mu2, sigma2 = 8, 2.0

# Range of x values
x = np.linspace(0, 15, 1000)

# Class-conditional probabilities
p_x_w1 = norm.pdf(x, mu1, sigma1)
p_x_w2 = norm.pdf(x, mu2, sigma2)

# Total probability P(x)
P_x = p_x_w1 * P_w1 + p_x_w2 * P_w2

# Posterior probabilities
P_w1_x = (p_x_w1 * P_w1) / P_x
P_w2_x = (p_x_w2 * P_w2) / P_x

# Decision boundary (P(w1|x) = P(w2|x))
decision_boundary = x[np.abs(P_w1_x - P_w2_x).argmin()]

# Plot posterior probabilities
plt.figure(figsize=(8, 5))
plt.plot(x, P_w1_x, label='$P(\omega_1|x)$', color='blue')
plt.plot(x, P_w2_x, label='$P(\omega_2|x)$', color='red')
plt.axvline(decision_boundary, color='black', linestyle='dashed', label='Decision Boundary')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.legend()
plt.title('Posterior Probabilities for Bayes Classifier')
plt.show()

# Shade decision regions
plt.figure(figsize=(8, 5))
plt.fill_between(x, P_w1_x, P_w2_x, where=(P_w1_x > P_w2_x), color='blue', alpha=0.2, label='Class ω1')
plt.fill_between(x, P_w1_x, P_w2_x, where=(P_w1_x < P_w2_x), color='red', alpha=0.2, label='Class ω2')
plt.axvline(decision_boundary, color='black', linestyle='dashed', label='Decision Boundary')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.legend()
plt.title('Decision Regions for Bayes Classifier')
plt.show()

# Similar approach for other questions...


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given priors
P_w1 = 0.3  # Prior probability of defective (ω1)
P_w2 = 0.7  # Prior probability of non-defective (ω2)

# Class-conditional probability densities (Gaussian distributions)
mu1, sigma1 = 4, 1.2  # Parameters for ω1 (defective)
mu2, sigma2 = 6, 1.5  # Parameters for ω2 (non-defective)

# Define the range for x values
x_values = np.linspace(0, 10, 1000)

# Compute the class-conditional densities
p_x_given_w1 = norm.pdf(x_values, mu1, sigma1)
p_x_given_w2 = norm.pdf(x_values, mu2, sigma2)

# Compute the total probability P(x)
P_x = p_x_given_w1 * P_w1 + p_x_given_w2 * P_w2

# Compute posterior probabilities using Bayes' theorem
P_w1_given_x = (p_x_given_w1 * P_w1) / P_x
P_w2_given_x = (p_x_given_w2 * P_w2) / P_x

# Find the decision boundary (where P_w1_given_x = P_w2_given_x)
decision_boundary_index = np.argmin(np.abs(P_w1_given_x - P_w2_given_x))
decision_boundary = x_values[decision_boundary_index]

# Plot the class-conditional densities
plt.figure(figsize=(8, 5))
plt.plot(x_values, p_x_given_w1, label=r'$p(x|\omega_1)$ (Defective)', linestyle='dashed', color='red')
plt.plot(x_values, p_x_given_w2, label=r'$p(x|\omega_2)$ (Non-Defective)', linestyle='dashed', color='blue')
plt.axvline(decision_boundary, color='black', linestyle='dotted', label="Decision Boundary")
plt.xlabel('Weight (x)')
plt.ylabel('Density')
plt.title('Class-Conditional Densities')
plt.legend()
plt.grid()
plt.show()

# Plot the posterior probabilities
plt.figure(figsize=(8, 5))
plt.plot(x_values, P_w1_given_x, label=r'$P(\omega_1|x)$ (Defective)', color='red')
plt.plot(x_values, P_w2_given_x, label=r'$P(\omega_2|x)$ (Non-Defective)', color='blue')
plt.axvline(decision_boundary, color='black', linestyle='dotted', label="Decision Boundary")
plt.xlabel('Weight (x)')
plt.ylabel('Probability')
plt.title('Posterior Probabilities')
plt.legend()
plt.grid()
plt.show()

# Print the decision boundary
print(f"Decision boundary occurs at x = {decision_boundary:.3f}")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given priors
P_w1 = 0.4  # Prior probability of High Risk (ω1)
P_w2 = 0.6  # Prior probability of Low Risk (ω2)

# Class-conditional probability densities (Gaussian distributions)
mu1, sigma1 = 550, 20  # Parameters for ω1 (High Risk)
mu2, sigma2 = 700, 25  # Parameters for ω2 (Low Risk)

# Define the range for x values (Credit Score)
x_values = np.linspace(450, 800, 1000)

# Compute the class-conditional densities
p_x_given_w1 = norm.pdf(x_values, mu1, sigma1)
p_x_given_w2 = norm.pdf(x_values, mu2, sigma2)

# Compute the total probability P(x)
P_x = p_x_given_w1 * P_w1 + p_x_given_w2 * P_w2

# Compute posterior probabilities using Bayes' theorem
P_w1_given_x = (p_x_given_w1 * P_w1) / P_x
P_w2_given_x = (p_x_given_w2 * P_w2) / P_x

# Loss function values
L_11, L_12 = 0, 2  # Loss for classifying as High Risk
L_21, L_22 = 5, 0  # Loss for classifying as Low Risk

# Compute the expected loss functions
R_alpha1_x = L_11 * P_w1_given_x + L_12 * P_w2_given_x  # Risk of classifying as High Risk
R_alpha2_x = L_21 * P_w1_given_x + L_22 * P_w2_given_x  # Risk of classifying as Low Risk

# Find the decision boundary (where R(α1|x) = R(α2|x))
decision_boundary_index = np.argmin(np.abs(R_alpha1_x - R_alpha2_x))
decision_boundary = x_values[decision_boundary_index]

# Plot the expected loss functions
plt.figure(figsize=(8, 5))
plt.plot(x_values, R_alpha1_x, label=r'$R(\alpha_1 | x)$ (High Risk)', color='red')
plt.plot(x_values, R_alpha2_x, label=r'$R(\alpha_2 | x)$ (Low Risk)', color='blue')
plt.axvline(decision_boundary, color='black', linestyle='dotted', label="Decision Boundary")
plt.xlabel('Credit Score (x)')
plt.ylabel('Expected Loss')
plt.title('Expected Loss Functions')
plt.legend()
plt.grid()
plt.show()

# Print the decision boundary
print(f"Decision boundary occurs at Credit Score x = {decision_boundary:.3f}")

# Discussion: Effect of increasing P(ω1) (High-Risk Applicants)
P_w1_new = 0.5  # Increased prior probability for High Risk
P_w2_new = 0.5  # Adjusted prior for Low Risk

# Recompute posterior probabilities with new priors
P_x_new = p_x_given_w1 * P_w1_new + p_x_given_w2 * P_w2_new
P_w1_given_x_new = (p_x_given_w1 * P_w1_new) / P_x_new
P_w2_given_x_new = (p_x_given_w2 * P_w2_new) / P_x_new

# Recompute the expected losses
R_alpha1_x_new = L_11 * P_w1_given_x_new + L_12 * P_w2_given_x_new
R_alpha2_x_new = L_21 * P_w1_given_x_new + L_22 * P_w2_given_x_new

# Find new decision boundary
decision_boundary_new_index = np.argmin(np.abs(R_alpha1_x_new - R_alpha2_x_new))
decision_boundary_new = x_values[decision_boundary_new_index]

# Print the new decision boundary
print(f"New decision boundary when P(High Risk) increases: Credit Score x = {decision_boundary_new:.3f}")

# Plot comparison before and after changing priors
plt.figure(figsize=(8, 5))
plt.plot(x_values, R_alpha1_x, '--', label=r'Original $R(\alpha_1 | x)$', color='red')
plt.plot(x_values, R_alpha2_x, '--', label=r'Original $R(\alpha_2 | x)$', color='blue')
plt.plot(x_values, R_alpha1_x_new, label=r'Updated $R(\alpha_1 | x)$', color='darkred')
plt.plot(x_values, R_alpha2_x_new, label=r'Updated $R(\alpha_2 | x)$', color='darkblue')
plt.axvline(decision_boundary, color='black', linestyle='dotted', label="Original Decision Boundary")
plt.axvline(decision_boundary_new, color='gray', linestyle='dotted', label="Updated Decision Boundary")
plt.xlabel('Credit Score (x)')
plt.ylabel('Expected Loss')
plt.title('Effect of Changing P(High Risk) on Decision Boundary')
plt.legend()
plt.grid()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given priors
P_w1 = 0.2  # Prior probability of Pedestrian (ω1)
P_w2 = 0.5  # Prior probability of Vehicle (ω2)
P_w3 = 0.3  # Prior probability of Empty Road (ω3)

# Class-conditional probability densities (Gaussian distributions)
mu1, sigma1 = 2, 0.5  # Parameters for ω1 (Pedestrian)
mu2, sigma2 = 5, 1.0  # Parameters for ω2 (Vehicle)
mu3, sigma3 = 10, 1.5  # Parameters for ω3 (Empty Road)

# Define the range for x values (Distance from LiDAR sensor)
x_values = np.linspace(0, 15, 1000)

# Compute the class-conditional densities
p_x_given_w1 = norm.pdf(x_values, mu1, sigma1)
p_x_given_w2 = norm.pdf(x_values, mu2, sigma2)
p_x_given_w3 = norm.pdf(x_values, mu3, sigma3)

# Compute the total probability P(x)
P_x = (p_x_given_w1 * P_w1) + (p_x_given_w2 * P_w2) + (p_x_given_w3 * P_w3)

# Compute posterior probabilities using Bayes' theorem
P_w1_given_x = (p_x_given_w1 * P_w1) / P_x
P_w2_given_x = (p_x_given_w2 * P_w2) / P_x
P_w3_given_x = (p_x_given_w3 * P_w3) / P_x

# Loss function values
L_11, L_12, L_13 = 0, 5, 10  # Loss for predicting Pedestrian
L_21, L_22, L_23 = 3, 0, 7  # Loss for predicting Vehicle
L_31, L_32, L_33 = 8, 6, 0  # Loss for predicting Empty Road

# Compute the expected loss functions
R_alpha1_x = L_11 * P_w1_given_x + L_12 * P_w2_given_x + L_13 * P_w3_given_x  # Risk of predicting Pedestrian
R_alpha2_x = L_21 * P_w1_given_x + L_22 * P_w2_given_x + L_23 * P_w3_given_x  # Risk of predicting Vehicle
R_alpha3_x = L_31 * P_w1_given_x + L_32 * P_w2_given_x + L_33 * P_w3_given_x  # Risk of predicting Empty Road

# Find the optimal decision for each x value (minimizing expected loss)
optimal_decision = np.argmin([R_alpha1_x, R_alpha2_x, R_alpha3_x], axis=0)

# Identify decision boundaries
boundary_indices = np.where(np.diff(optimal_decision) != 0)[0]
decision_boundaries = x_values[boundary_indices]

# Plot posterior probabilities
plt.figure(figsize=(8, 5))
plt.plot(x_values, P_w1_given_x, label=r'$P(\omega_1|x)$ (Pedestrian)', color='red')
plt.plot(x_values, P_w2_given_x, label=r'$P(\omega_2|x)$ (Vehicle)', color='blue')
plt.plot(x_values, P_w3_given_x, label=r'$P(\omega_3|x)$ (Empty Road)', color='green')
for boundary in decision_boundaries:
    plt.axvline(boundary, color='black', linestyle='dotted', label="Decision Boundary")
plt.xlabel('Distance from LiDAR Sensor (x)')
plt.ylabel('Probability')
plt.title('Posterior Probabilities')
plt.legend()
plt.grid()
plt.show()

# Plot expected loss functions
plt.figure(figsize=(8, 5))
plt.plot(x_values, R_alpha1_x, label=r'$R(\alpha_1 | x)$ (Predict Pedestrian)', color='red')
plt.plot(x_values, R_alpha2_x, label=r'$R(\alpha_2 | x)$ (Predict Vehicle)', color='blue')
plt.plot(x_values, R_alpha3_x, label=r'$R(\alpha_3 | x)$ (Predict Empty Road)', color='green')
for boundary in decision_boundaries:
    plt.axvline(boundary, color='black', linestyle='dotted', label="Decision Boundary")
plt.xlabel('Distance from LiDAR Sensor (x)')
plt.ylabel('Expected Loss')
plt.title('Expected Loss Functions')
plt.legend()
plt.grid()
plt.show()

# Print the decision boundaries
print(f"Decision boundaries occur at x-values: {decision_boundaries}")

# Effect of increasing penalty for misclassifying a pedestrian
L_12_new, L_13_new = 8, 15  # Increased penalties for misclassification
R_alpha1_x_new = L_11 * P_w1_given_x + L_12_new * P_w2_given_x + L_13_new * P_w3_given_x

# Recompute optimal decision
optimal_decision_new = np.argmin([R_alpha1_x_new, R_alpha2_x, R_alpha3_x], axis=0)
boundary_indices_new = np.where(np.diff(optimal_decision_new) != 0)[0]
decision_boundaries_new = x_values[boundary_indices_new]

# Print updated decision boundaries
print(f"New decision boundaries after increasing pedestrian misclassification penalty: {decision_boundaries_new}")

# Plot updated expected loss functions
plt.figure(figsize=(8, 5))
plt.plot(x_values, R_alpha1_x_new, '--', label=r'Updated $R(\alpha_1 | x)$', color='darkred')
plt.plot(x_values, R_alpha2_x, label=r'$R(\alpha_2 | x)$', color='blue')
plt.plot(x_values, R_alpha3_x, label=r'$R(\alpha_3 | x)$', color='green')
for boundary in decision_boundaries_new:
    plt.axvline(boundary, color='gray', linestyle='dotted', label="Updated Decision Boundary")
plt.xlabel('Distance from LiDAR Sensor (x)')
plt.ylabel('Expected Loss')
plt.title('Effect of Increasing Penalty for Pedestrian Misclassification')
plt.legend()
plt.grid()
plt.show()

