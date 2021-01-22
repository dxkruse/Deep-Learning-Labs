# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Requests a word from user input 
# Creates new_string using only the letters from indices 1:4
# Reverses new_string before printing to the console
# =============================================================================
string = input("Please enter a word: ")
new_string = string[1:4]
new_string = new_string[::-1]
print(new_string)

# =============================================================================
# Requests two numbers, x and y, from user input and immediately splits them
# and converts them to integers. This is required as the input() function 
# returns a string. Syntax found at https://www.geeksforgeeks.org/input-multiple-values-user-one-line-python/ 
# Prints the value of x to the power of y to the console
# =============================================================================
x,y = [int(x) for x in input("Please enter two integers separated by a space: ").split()]
print(x**y)


# =============================================================================
# Requests a sentence from user input
# Replaces all instances of python with pythons before printing to the console
# =============================================================================
sentence = input("Please enter a sentence that uses the word 'python': ")
new_sentence = sentence.replace("python", "pythons")
print(new_sentence)

