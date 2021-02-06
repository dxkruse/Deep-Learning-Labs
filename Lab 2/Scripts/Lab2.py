# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:32:32 2021

@author: dxkru
"""

# Problem 1

# Request input, splits the input using spaces as the delimiter, uses map() 
# function to apply the float() to each item in the map, then converts to a list
# and assigns to variable x_ft.  
x_ft = list(map(float,input("Input numbers separated by a space: ").split()))
# Loops through x_ft list and multiplies each item by a conversion factor of
# 30.48 cm/ft
x_cm = [i*30.48 for i in x_ft]
print(x_cm)

# Problem 2

# Requests user input and converts from string to integer
x = int(input("Input any positive integer: "))
# Initialize steps counter
steps = 0
# Loop while value of x is not zero
while x != 0:
    # If remainder is 0, x is even so it is divided by two and steps counter
    # increased by one
    if x % 2 == 0:
        x /= 2
        steps += 1
    # Else if remainder is 1, x is odd so one is subtracted and steps counter
    # is increased by one
    elif x % 2 == 1:
        x -= 1
        steps += 1
print(x,steps)        
    
# Problem 3

# Open text file with read and write priveleges
text = open("Prob3.txt", "r+")
# Initialize an empty dictionary
d = dict()
# Loop through lines in text
for line in text:
    # Split line into separate words and store in words variable
    words = line.split()
    # Loop through words
    for word in words:
        # If the word is in the dictionary already, add one to the word count
        # for that word
        if word in d:
            d[word] = d[word] + 1
        # Else if word has not been added to the dictionary, add it and set
        # its word count to one
        else:
            d[word] = 1
            

# Loop through dictionary using words as keys
for key in list(d.keys()):
    # Set temp_str variable (Not necessary, but done for debugging and ease of
    # reading). Prints the string to the console and writes it to the text file
    # on a new line
    temp_str = key + ": " + str(d[key])
    print(temp_str)
    text.write("\n" + temp_str)
    
# Closes the text file
text.close()