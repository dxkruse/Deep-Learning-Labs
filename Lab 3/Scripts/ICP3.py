# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:43:18 2021

@author: Dietrich
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os



#%% Problem 1

# Create Employee Class
class Employee:
    # Initialize total employee count
    total_count = 0
    
    # Initialize constructor with all inputs
    def __init__(self, name, family, salary, department, hourly_pay, weekly_hours):
        self.name = name
        self.family = family
        self.hourly_pay = hourly_pay
        self.weekly_hours = weekly_hours
        self.salary = salary
        self.department = department
    
    # Create increment function to increment total employee count
    def increment(self):
        Employee.total_count += 1

    # Calculate salary function
    def get_avg_salary(self):
        salary = self.hourly_pay*self.weekly_hours*52
        print(salary)
        return salary

# Create Fulltime Employee class that inherits Employee class        
class Fulltime_Employee(Employee):
    def __init__(self, name, family, salary, department, hourly_pay, num_hours):
        Employee.__init__(self, name, family, salary, department, hourly_pay, num_hours)
        # Initialize fulltime total employee count
        self.fulltime_total = 0
    # Update increment function to include fulltime_total
    def increment(self):
        Employee.total_count += 1
        self.fulltime_total += 1

# Create instances of two employees and call increment function
emp1 = Employee("Dietrich", "Kruse Family", 100000, "Gaming Department", 25, 15)
emp1.increment()
emp2 = Fulltime_Employee("Dietrich", "Kruse Family", 100000, "Gaming Department", 25, 40)
emp2.increment()

# Print employee totals
print(emp1.__class__.total_count)
print(emp2.fulltime_total)

# Call salary function
emp1.get_avg_salary()
emp2.get_avg_salary()


#%% Problem 2

# Initialize URL
url = "https://en.wikipedia.org/wiki/Deep_learning"

# Save website contents to html variable
html = requests.get(url)

# Parse all html content into bsObj variable
bsObj = BeautifulSoup(html.content, "html.parser")

# Print title of page
print(bsObj.title.string)

# Set file name and open file
file_name = "links.txt"
if not os.path.exists(file_name):
    print("Creating file " + file_name)
    file = open('links.txt','a+',encoding='utf-8')
    
# Iterate through all links and write each one to the desired file, adding a new
# line after each link.    
for link in bsObj.find_all('a'):
    file.write(str(link.get('href')) + '\n')

# Close the file
file.close()


#%% Problem 3

#Create array of random numbers betwen of shape (1,20)
x = 20 * np.random.random((1,20))
print(x)
# Reshape to (4,5)
x = np.reshape(x, (4,5))
print(x)
# Get values of row maximums
row_maxes = x.max(axis=1).reshape(-1,1)
print(row_maxes)
# Replace maximum values in each row with 0
x = np.where(x != row_maxes, x, 0)
print(x)

        
    