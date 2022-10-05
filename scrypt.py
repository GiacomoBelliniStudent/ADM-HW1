# Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")
    
# Python If-Else
    
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n%2 != 0:
    print("Weird")
elif 2<=n<=5:
    print("Not Weird")
elif 6<=n<=20:
    print("Weird")
else:
    print("Not Weird")

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
if 1 <= a <= 10**10 and 1 <= b <= 10**10:
    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print(a//b)
print(a/b)

# Loops

if __name__ == '__main__':
    n = int(input())

for x in range(0,n):
    print(x**2)

# Write a function

def is_leap(year):
    leap = False
    
    # Write your logic here
    if(1900 <= year <= 10 ** 5):
        if year%4 == 0:
            leap = True
        if year%100 == 0:
            leap = False
        if year%400 == 0:
            leap = True
            
    return leap

# Print Function

if __name__ == '__main__':
    n = int(input())
for x in range(1,n+1):
    print(x, end="")

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i + j+ k != n])

# Find the Runner-Up Score! 

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    sortedList = list(set(arr))
    sortedList.sort()
    print(sortedList[-2])

# Nested Lists

if __name__ == '__main__':
    records=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name,score]);

sortedRecords = sorted(list(set([x[1] for x in records])))
worstStudents = []
for x in records:
    if x[1] == sortedRecords[1]:
        worstStudents.append(x[0])
for name in sorted(worstStudents):
    print(name)

# Finding the percentage

def Average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
average = Average(student_marks[query_name])
print("{:.2f}".format(average))

# Tuples 

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

print(hash(tuple(integer_list)))

# Lists

if __name__ == '__main__':
    array = list()
    N = int(input())
    for k in range(N):
        command, *rest = input().split()
        params = list(map(int, rest))
        if len(params)==2:
            q=params[0]
            w=params[1]
        elif len(params)==1:
            q=params[0]
        if command =='insert':
            array.insert(q,w)
        elif command == 'append':
            array.append(q)
        elif  command == 'remove':
            array.remove(q)
        elif command =='print':
            print(array)
        elif command == 'reverse':
            array.reverse()
        elif command =='pop':
            array.pop()
        elif command == 'sort':
            array.sort()

# sWAP cASE

def swap_case(s):
    return s.swapcase()

# String Split and Join

def split_and_join(line):
    # write your code here
    return line.replace(" ", "-")

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    # Write your code here
    print("Hello " + first + " " + last + "! You just delved into python.")

# Mutations

def mutate_string(string, position, character):
    return string[0:position] + character + string[position+1:]

# Find a string

import re

def count_substring(string, sub_string):
    counter = 0
    for i in range(len(string)):
       if string[i:].startswith(sub_string):
          counter += 1
    return counter

# String Validators

if __name__ == '__main__':
    s = input()
    
print(any(map(str.isalnum,s)))
print(any(map(str.isalpha,s)))
print(any(map(str.isdigit,s)))
print(any(map(str.islower,s)))
print(any(map(str.isupper,s)))

# Text Alignment

# Enter your code here. Read input from STDIN. Print output to STDOUT
thickness = int(input())

letter = "H"

    
for i in range(thickness):
    print((letter*i).rjust(thickness-1)+letter+(letter*i).ljust(thickness-1))

for i in range(thickness+1):
    print((letter*thickness).center(thickness*2)+(letter*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((letter*thickness*5).center(thickness*6))   

for i in range(thickness+1):
    print((letter*thickness).center(thickness*2)+(letter*thickness).center(thickness*6))   

for i in range(thickness):
    print(((letter*(thickness-i-1)).rjust(thickness)+letter+(letter*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))  

# Text Wrap

def wrap(string, max_width):
  wrapped = textwrap.wrap(string, max_width)
  return '\n'.join(wrapped)

# Designer Door Mat

# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = map(int, input().split())
for i in range(1, N, 2):
    print((i * ".|.").center(M,"-"))
print("WELCOME".center(M, "-"))
for i in range(N-2, -1, -2):
    print((i * ".|.").center(M,"-"))

# String Formatting

def print_formatted(number):
    # your code goes here
    binary = bin(number).replace("0b","")
    for i in range(1,n+1):
        decimal = (len(binary)-len(str(i)))*" "+str(i)
        octal =  (len(binary)-len(str(oct(i).replace("0o",""))))*" "+str(oct(i)           .replace("0o",""))
        hexa = (len(binary)-len(hex(i).replace("0x","").upper()
        ))*" "+hex(i).replace("0x","").upper()
        bina =  (len(binary)-len(bin(i).replace("0b","")))*" "+bin(i).replace("0b","")
        print(decimal,octal,hexa,bina)
        
# Alphabet Rangoli

import string

def print_rangoli(size):
    # your code goes here
    array = []
    design = string.ascii_lowercase
    for i in range(size):
        s = "-".join(design[i:size])
        array.append((s[::-1]+s[1:]).center(4*size-3, "-"))
     
    print('\n'.join(array[:0:-1]+array))
    
# Capitalize!

# Complete the solve function below.
def solve(s):
    splitted = s.split()
    for word in splitted:
        s = s.replace(word, word.capitalize())
    return s

# The Minion Game

def minion_game(string):
    # your code goes here
    scores = {'Stuart': 0, 'Kevin': 0}
    for i in range(len(string)):
        if string[i] in 'AEIOU':
            scores['Kevin'] += len(string) - i
        else:
            scores['Stuart'] += len(string) - i
    if scores['Stuart'] > scores["Kevin"]:
        print("Stuart " + str(scores["Stuart"]))
    elif scores["Kevin"] > scores["Stuart"]:
        print("Kevin " + str(scores["Kevin"]))
    else:
        print("Draw")
        
# Merge the Tools!

def merge_the_tools(string, k):
    # your code goes here
    array = [string[i:i + k] for i in range(0, len(string), k)]
    for x in array:
        merged = set(x)
        res = ""
        arrayList = list(merged)
        for char in arrayList:
           res += char
        print(res) 
        
# Introduction to Sets

def average(array):
    # your code goes here
    return sum(set(array))/len(set(array))

# Symmetric Difference

m = int(input())
M = list(map(int, input().strip().split()))
n = int(input()), 
N = list(map(int, input().strip().split()))

firstSet = set(M)
secondSet = set(N)

for i in sorted(firstSet ^ secondSet):
    print(i)
    
# No Idea!

# Enter your code here. Read input from STDIN. Print output to STDOUT
happiness = 0
integers = list(map(int, input().strip().split()))
n = integers[0]
m = integers[1]
N = list(map(int, input().strip().split()))
A = set(map(int, input().strip().split()))
B = set(map(int, input().strip().split()))

for x in N:
    if x in A:
        happiness += 1
    elif x in B:
        happiness -= 1
print(happiness)

# Set .add() 

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
countries = set()
for i in range(n):
    countries.add(input().strip())
print(len(countries))

# Set .union() Operation

n = input()
a = set(map(int, input().split()))
n = input()
b = set(map(int, input().split()))

c = a.union(b)
print(len(c))

# Set .intersection() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = input()
a = set(map(int, input().split()))
n = input()
b = set(map(int, input().split()))

c = a.intersection(b)
print(len(c))

# Set .difference() Operation

n = input()
a = set(map(int, input().split()))
n = input()
b = set(map(int, input().split()))

c = a.difference(b)
print(len(c))

# Set .symmetric_difference() Operation

n = input()
a = set(map(int, input().split()))
n = input()
b = set(map(int, input().split()))

c = a.symmetric_difference(b)
print(len(c))

# Set Mutations

# Enter your code here. Read input from STDIN. Print output to STDOUT
Ael = int(input())
A = set(map(int,input().split()))
N = int(input())

for i in range(N):
    n = input().split()
    array = set(map(int,input().split()))
    
    if n[0] == 'update':
        A.update(array)
    elif n[0] == 'intersection_update':
        A.intersection_update(array)
    elif n[0] == 'symmetric_difference_update':
        A.symmetric_difference_update(array)
    elif n[0] == 'difference_update':
        A.difference_update(array)
        
print(sum(A))

# The Captain's Room 

rooms = list(map(int,input().split()))

a = set()
b = set()

for i in rooms:
    if i in a:
        b.add(i)
    else:
        a.add(i)
print(a.difference(b).pop())

# Check Subset

n = int(input())

for i in range(n):
    a = int(input())
    A = set(map(int,input().split()))

    b = int(input())
    B = set(map(int,input().split()))

    if len(A - B) == 0:
        print("True")
    else:
        print("False")
        
# Check Strict Superset

A = set(map(int,input().split()))

n = int(input())
res = "True"
for i in range(n):
    B = set(map(int,input().split()))
    if len(B - A) != 0:
        res = "False"
        
print(res)

# collections.Counter()

n = int(input())

shoeSizes = list(map(int,input().split()))
customers = int(input())
res = 0
for customer in range(customers):
    values = list(map(int,input().split()))
    if values[0] in shoeSizes:
        shoeSizes.remove(values[0])
        res += values[1]
        
print(res)

# DefaultDict Tutorial

from collections import defaultdict

d = defaultdict(list)

n,m = map(int, input().split())

for i in range(n):
    d[input()].append(str(i+1))
for i in range(m):
    print(" ".join(d[input()]) or -1)
    
# Collections.namedtuple()

from collections import namedtuple
n = int(input())
student = namedtuple('Student',input().split())
res = 0

for i in range(n):
    MARKS, IDS, NAME, CLASS = input().split() 
    stud = student(MARKS, IDS, NAME, CLASS)
    res += int(stud.MARKS)
print(res/n)

# Collections.OrderedDict()

from collections import OrderedDict

ordered_dictionary = OrderedDict()

n = int(input())

for i in range(n):
    item, space, price = input().rpartition(" ")
    ordered_dictionary[item] = ordered_dictionary.get(item,0) + int(price)

for item, price in ordered_dictionary.items():
    print(item,price)
    
# Word Order

from collections import OrderedDict
n = int(input())

ordered_dict = OrderedDict()

for i in range(n):
    word = input()
    ordered_dict[word] = ordered_dict.get(word,0) + 1
    
print(len(ordered_dict.keys()))
for word, count in ordered_dict.items():
    print(count, end=" ")
    
# Collections.deque()

from collections import deque

n = int(input())
deque = deque()

for i in range(n):
    comm = input().split(" ")
    if comm[0] == 'append':
        deque.append(comm[1])
    elif comm[0] == 'appendleft':
        deque.appendleft(comm[1])
    elif comm[0] == 'pop':
        deque.pop()
    elif comm[0] == 'popleft':
        deque.popleft()

for x in deque:
    print(x, end=" ")

# Piling Up!

from collections import deque
testCases = int(input())

for _ in range(testCases):
    n = int(input())
    numDeque = deque(list(map(int,input().split())))
    accept = "Yes"
    num = []
    for i in range(n-1):
        left = numDeque.popleft()
        right = numDeque.pop()
        if left > right and (len(num)==0 or left <= num[-1]) :
            numDeque.append(right)
            num.append(left)
        elif len(num)==0 or right <= num[-1]:
            numDeque.appendleft(left)
            num.append(right)
        else:
            accept = "No"
            break
    print(accept)
    
# Company Logo

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    s = input()
    d=dict()
    for c in s:
        d[c] = d.get(c,0) + 1

res = {val[0] : val[1] for val in sorted(d.items(), key = lambda x: (-x[1], x[0]))}
for x,y in list(res.items())[0:3]:
    print(x, y)
    
# Calendar Module

import calendar
import datetime

month, day, year = map(int,input().split())

print(calendar.day_name[datetime.date(year, month, day).weekday()].upper())

# Time Delta

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
   t1 = datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")
   t2 = datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
   return abs(int((t1-t2).total_seconds()))
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = str(time_delta(t1, t2))
        
# Exceptions

n = int(input())

for _ in range(n):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as f:
        print("Error Code:", f)

        fptr.write(delta + '\n')

    fptr.close()
    
# Incorrect Regex

import re

n = int(input())

for _ in range(n):
    try:
        word= input()
        regexp = re.compile(word)
        if regexp.search(word):
            print('True')
    except Exception as e:
          print("False")  
          
# Zipped!

n,x = map(int,input().split())
array = []

for _ in range(x):
    array += [list(map(float,input().split()))]
    
c = zip(*array)
for s in c:
    print(sum(s)/x)
    
# Athlete Sort

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    ranking=sorted(arr,key=lambda row:row[k])
    for i in range(len(ranking)):
        for j in range(len(ranking[i])):
            print(ranking[i][j], end=' ')
        print()

# ginortS

def rules(s):
    if s.islower():
        return ord(s)
    elif s.isupper():
        return ord(s)*10
    elif s in "13579":
        return ord(s)*100
    else:
        return ord(s)*1000

s = input()

print(*sorted(s, key= lambda k:(rules(k))),sep='')

# Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    if n<2:
        return range(n)
    fibo = [0,1]
    for i in range(n-2):
        fibo.append(fibo[i+1] + fibo[i])
    return fibo

# Detect Floating Point Number

import re
t = int(input())

regex = re.compile('^[-+]?[0-9]*\.[0-9]+$')
for _ in range(t):
    n = input()
    if regex.match(n):
        print('True')
    else:
        print('False')
        
# Re.split()

regex_pattern = r"[,.]"	# Do not delete 'r'.

# Group(), Groups() & Groupdict()

import re

res = re.findall(r'([A-Za-z0-9])\1+',input())

if not res:
    print(-1)
else:
    print(res[0])
    
# Re.findall() & Re.finditer()

import re
consonants = 'qwrtypsdfghjklzxcvbnm'
vowels = 'aeiou'

res = re.findall(r'(?<=['+consonants+'])(['+vowels+']{2,})(?=['+consonants+'])',input(),flags=re.I)

print('\n'.join(res or ['-1']))

# Re.start() & Re.end()

import re

s = input()
k = input()

res = re.finditer(r'(?=('+k+'))',s)

find = False
for i in res:
    find = True
    print((i.start(1),i.end(1)-1))
if find == False:
    print((-1,-1))
    
# Regex Substitution

import re

n = int(input())

for _ in range(n):
    s = input()
    s = re.sub("(?<= )(&&)(?= )", "and", s)
    s = re.sub("(?<= )(\|\|)(?= )", "or", s)
    print(s)
    
# Validating phone numbers

import re
n= int(input())

for _ in range(n):
    if re.match(r'^[789]\d{9}$',input()):
        print("YES")
    else:
        print("NO")
        
# Validating and Parsing Email Addresses

import email.utils
import re

n = int(input())

for _ in range(n):
    s = input()
    parseString = email.utils.parseaddr(s)

    if re.search("^[a-z][\w.-]+@[a-z]+\.[a-z]{1,3}$",parseString[-1],re.I):
        print(s)
        
# Hex Color Code

import re

for _ in range(int(input())):
    css = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})',input())
    if css:
        print(*css, sep="\n")
        
# HTML Parser - Part 1

from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])
    def handle_endtag(self, tag):
        print ('End   :', tag)
    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])
parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

# TML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        prefix = '\n' in data and 'Multi-line Comment' or 'Single-line Comment'
        print('>>> {0}\n{1}'.format(prefix, data))
    def handle_data(self, data):
        if data is not '\n':
            print('>>> Data\n{0}'.format(data))
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])
        
  
html = ""       
for i in range(int(input())):
    html += input()
parser = MyHTMLParser()
parser.feed(html)

# Validating UID 

import re

t = int(input())

for _ in range(t):
    s = input()
    s = ''.join(sorted(s))
    
    if re.search(r"[A-Z]{2}",s) and re.search(r"\d{3}",s) and not re.search(r"[^A-Za-z0-9]",s) and not re.search(r"(\w)\1",s) and len(s) == 10:
        print("Valid")
    else:
        print("Invalid")
        
# Validating Credit Card Numbers

import re
for _ in range(int(input())):
    s = input()
    if(re.match(r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$", s) and not re.search(r"([\d])\1\1\1", s.replace("-", ""))):
        print("Valid")
    else:
        print("Invalid")
        
# Validating Postal Codes

regex_integer_in_range = r"^[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.

# Matrix Script

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = [''] * (n*m)

for i in range(n):
    matrix_item = input()
    for j in range(m):
        matrix[i+(j*n)]=matrix_item[j]
decoded_str = ''.join(matrix)
final_decoded_str = re.sub(r'(?<=[A-Za-z0-9])([ !@#$%&]+)(?=[A-Za-z0-9])',' ',decoded_str)
print(final_decoded_str) 

# Validating Email Addresses With a Filter 

import re 

def fun(s):
    # return True if s is a valid email, else return False
    pattern = re.compile("^[\\w-]+@[0-9a-zA-Z]+\\.[a-z]{1,3}$")
    return pattern.match(s)

# Reduce Function

def product(fracs):
    t = reduce(lambda x,y : x * y, fracs)
    return t.numerator, t.denominator

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        return map(f,sorted(people, key = lambda x: int(x[2])))
    return inner

# Arrays

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr[::-1], float)

# Shape and Reshape

import numpy as np

n = list(map(int,input().split()))
print(np.reshape(n,(3,3)))

# Transpose and Flatten

import numpy as np

array = list()
n,m = map(int,input().split())

for _ in range(n):
    array.append(list(map(int,input().split())))

print(np.transpose(np.array(array)))
print(np.array(array).flatten())

# Concatenate

import numpy as np

n,m,p = map(int,input().split())

first = np.array([input().split() for i in range(n)], int)
second = np.array([input().split() for i in range(m)], int)

print(np.concatenate((first,second),axis = 0))

# Zeros and Ones

import numpy as np
dim = tuple([int(i) for i in input().split()])
print(np.zeros(dim,dtype = np.int))
print(np.ones(dim,dtype = np.int))

# Eye and Identity

import numpy
numpy.set_printoptions(legacy="1.13")

n,m = map(int, input().split())
print(numpy.eye(n,m,k=0))

# Array Mathematics

import numpy

n,m = map(int, input().split())

A = numpy.array([list(map(int, input().split())) for i in range(n)])
B = numpy.array([list(map(int, input().split())) for i in range(n)])

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')

A = numpy.array(input().split(), float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())):
    s1 = input().split()
    if s1[0] == 'pop':
        s.pop()
    elif s1[0] == 'remove':
        s.remove(int(s1[1]))
    elif s1[0] == 'discard':
        s.discard(int(s1[1]))
print(sum(s))

# Sum and Prod

import numpy

n,m = map(int,input().split())

arr = numpy.array([input().split() for _ in range(n)],int)

print(numpy.prod(numpy.sum(arr,axis=0)))

# Min and Max

import numpy

n,m = map(int,input().split())

arr = numpy.array([input().split() for _ in range(n)],int)

print(numpy.max(numpy.min(arr,axis=1)))

# Mean, Var, and Std

import numpy

n,m = map(int,input().split())

arr = numpy.array([input().split() for _ in range(n)],int)

print(numpy.mean(arr,1),numpy.var(arr,0),round(numpy.std(arr,None),11),sep="\n")

# Dot and Cross

import numpy

n = int(input())

arr1 = numpy.array([input().split() for _ in range(n)],int)
arr2 = numpy.array([input().split() for _ in range(n)],int)

print(numpy.dot(arr1,arr2))

# Inner and Outer

import numpy

A = numpy.array(list(input().split()),int)
B = numpy.array(list(input().split()),int)

print(numpy.inner(A, B),numpy.outer(A,B),sep="\n")

# Polynomials

import numpy

print(numpy.polyval(list(map(float,input().split())), int(input())))

# Linear Algebra

import numpy
numpy.set_printoptions(legacy='1.13')

a = list()

for _ in range(int(input())):
    a.append(list(map(float,input().split())))
    
print(numpy.linalg.det(a))

# Birthday Cake Candles

import math
import os
import random
import re
import sys
from collections import Counter

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    counts = Counter(candles)
    return counts[max(candles)]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    if x1 > x2 and v1 > v2:
        return("NO")
    elif x1 < x2 and v1 < v2:
        return("NO")
    elif v1 == v2:
        return("NO")
    elif (x2-x1)%(v1-v2) == 0 or (x2-x1)%(v2-v1) == 0:
        return("YES")
    else:
        return("NO")

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    # Write your code here
    init = 5
    cum = 0
    for _ in range(n):
        liked = init//2
        cum += liked
        init = liked * 3
        
    return cum

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
# Recursive Digit Sum   

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k

def calculate(p):
    if(len(p)==1):
        return p
    else:
        p = map(int, list(p))
        return calculate(str(sum(p)))

def superDigit(n, k):
    # Write your code here
    n = map(int, list(n))
    return calculate(str(sum(n)*k))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    # Write your code here
    t = arr[-1]
    index = n-2
    
    while(t < arr[index]) and index >= 0:
        arr[index+1] = arr[index]
        print(*arr)
        arr[index] = t
        index = index -1
        
    arr[index +1] = t
    print(*arr)
    

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion Sort - Part 2

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(index, arr):
    t = arr[index]
    
    for idx in range(index-1, -1, -1):
        if arr[idx] > t:
            arr[idx+1] = arr[idx]
        else:
            arr[idx+1] = t
            break
    if arr[0] > t:
        arr[0] = t
    

def insertionSort2(n, arr):
    # Write your code here
    for index in range(1, len(arr)):
        insertionSort1(index, arr)
        print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
