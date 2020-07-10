import sys


class Branch:
    def __init__(self, branch_tokens):
        '''This method initialises the attributes common among all branches:
        Name, Type, Source, Destination, and Value.'''
        self.name = branch_tokens[0]
        dict_object_type = {"R": "Resistor",
                            "L": "Inductor",
                            "C": "Capacitor",
                            "V": "Independent Voltage Source",
                            "I": "Independent Current Source",
                            "E": "Voltage Controlled Voltage Source",
                            "G": "Voltage Controlled Current Source",
                            "H": "Current Controlled Voltage Source",
                            "F": "Current Controlled Current Source"}

        self.type = dict_object_type[self.name[0]]
        self.source = branch_tokens[1]
        self.dest = branch_tokens[2]
        self.value = float(branch_tokens[-1])

    def __str__(self):
        ''' Printing the branch will give all the details 
        in a human understandable format'''
        part_one = ("Branch  {}\n".format(self.name))
        part_two = ("Source {} Destination {}\n".format(self.source, self.dest))
        part_three = ("type {} Value {}\n".format(self.type, self.value))
        return (part_one+part_three+part_two)


class Voltage_Controlled_Branch(Branch):
    def __init__(self, branch_tokens):
        Branch.__init__(self, branch_tokens)
        self.control_source = branch_tokens[3]
        self.control_destination = branch_tokens[4]

    def __str__(self):
        part_one = Branch.__str__(self)
        part_two = "Control_Souce {} Control_Dest {}\n".format(
            self.control_source, self.control_destination)
        return (part_one+part_two)


class Current_Controlled_Branch(Branch):
    def __init__(self, branch_tokens):
        Branch.__init__(self, branch_tokens)
        self.control_name = branch_tokens[3]

    def __str__(self):
        part_one = Branch.__str__(self)
        part_two = "Control_Name {} \n".format(self.control_name)
        return (part_one+part_two)


def extract_file_name():
    '''This function takes the file name from CLI'''
    try:
        file_name = sys.argv[1]
    except IndexError:
        print("Netlist File name not supplied, unable to proceed.")
        file_name = None
    return file_name


def validate_file(file_name):
    '''This Function reads the file and retuns the lines'''
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            return lines
    except FileNotFoundError:
        print("File Did Not Exist, Quitting program")
        quit()


def validate_circuit_existence(file_name):
    '''This function validates that the file has a circuit in it.'''
    lines = validate_file(file_name)
    lines = [i.strip() for i in lines]

    try:
        start = lines.index(".circuit")
        end = lines.index(".end")

        if start > end:
            raise ValueError
        circuit_lines = lines[start+1:end]
        return circuit_lines
    except ValueError:
        print("Either .circuit or .end was missing or .circuit was after .end ")
        quit()


def clean_branch(branch):
    '''This Function removes extra spaces and the comments from the branch '''
    branch_tokens = branch.split()
    branch_tokens = list(filter(lambda a: a != " ", branch_tokens))
    for i in range(len(branch_tokens)):
        if branch_tokens[i][0] == "#":
            flag = 1
            break
    else:
        flag = 0
    if flag == 1:
        del branch_tokens[i:]
    return branch_tokens


def parse(branch):
    '''Function decides what type of branch is used, 
    and also throws error if branch is invalid'''
    branch_tokens = clean_branch(branch)
    if branch_tokens[0][0] in "RLCAV":
        parsed_branch = Branch(branch_tokens)
    elif branch_tokens[0][0] in "EG":
        parsed_branch = Voltage_Controlled_Branch(branch_tokens)
    elif branch_tokens[0][0] in "FH":
        parsed_branch = Current_Controlled_Branch(branch_tokens)
    else:
        print("Invalid name", end=" ")
        print(branch_tokens[0], "Please Check the Netlist and try again")
        quit()
    return branch_tokens, parsed_branch


def branch_not_comment(branch):
    '''If a line is empty or just a comment, this function removes it immediately'''
    if branch == "" or branch[0] == '#':
        flag = 0
    else:
        flag = 1
    return flag


def analyse_circuit(circuit_lines):
    '''Function goes through the circuit line by line and parses it.'''
    list_branches = []
    for branch in circuit_lines:
        if branch_not_comment(branch):
            branch_tokens, parsed_branch = parse(branch)
            list_branches.append([branch_tokens, parsed_branch])
    return list_branches


def circuit_reverse_print(list_branches):
    '''Function reverse prints the circuit'''
    len_circuit = len(list_branches)
    for i in range(len_circuit-1, -1, -1):
        for j in list_branches[i][0][::-1]:
            print(j, end=" ")
        print()


def main():
    file_name = extract_file_name()
    if file_name is not None:
        circuit_lines = validate_circuit_existence(file_name)
        circuit_processed_format = analyse_circuit(circuit_lines)
        circuit_reverse_print(circuit_processed_format)


if __name__ == "__main__":
    main()