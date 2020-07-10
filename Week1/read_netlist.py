import sys
from sys import argv, exit

#with open(sys.argv[1], 'r') as my_file:
#   print("The circuit read from file is \n")
#   print(my_file.read())

CIRCUIT = '.circuit'
END = '.end'

print("The number of arguments received by %s = %d" % (sys.argv[0], len(sys.argv)))
print("\nThe modified and reversed circuit is ")
#file = input("Type the filename ")

if len(argv) != 2:
    print('\nPlease type the filename after %s' % argv[0])
    exit()

try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1; end = -2
        for line in lines:              # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end:                # validating circuit block
            print('Invalid circuit definition')
            exit(0)

        for line in reversed([' '.join(reversed(line.split('#')[0].split())) for line in lines[start+1:end]]):
            print(line)                 # print output

except IOError:
    print('Invalid file')
    exit()

    
"""
#if len(sys.argv) < 2:
#   print("Oops! That was not a valid filename. Try again...")
#try:
#      C = float(sys.argv[1])
#except Exception:
#    print("Oops! That was not a valid filename. Try again...")"""
   

