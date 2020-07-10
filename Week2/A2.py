import sys
import numpy as np
import math
import pprint
from math import pi
class Node:
    def __init__(self,name):
        self.name = name
        self.from_list = []
        self.to_list = []

    def add_branch(self,branch):
        if branch.source == self.name:
            self.from_list.append(branch)
        elif branch.dest == self.name:
            self.to_list.append(branch)


class Branch:
    def __init__(self, branch_tokens,aclist=[]):
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
        if self.type not in ["Independent Voltage Source","Independent Current Source"]: 
            self.value = float(branch_tokens[-1])


class Source_Branch(Branch):
    def __init__(self, branch_tokens,aclist):
        Branch.__init__(self, branch_tokens)
        self.current_type = branch_tokens[3]
        if self.current_type =="ac":
            self.value = float(branch_tokens[4])
            self.phase = float(branch_tokens[5])
            for elem in aclist:
                if elem[0]==self.name:
                    self.freq = elem[1]
                    break
            else:
                print("AC Source but frequency not supplied")
                quit()
        else:
            self.value = float(branch_tokens[-1])



class Voltage_Controlled_Branch(Branch):
    def __init__(self, branch_tokens):
        Branch.__init__(self, branch_tokens)
        self.control_source = branch_tokens[3]
        self.control_destination = branch_tokens[4]


class Current_Controlled_Branch(Branch):
    def __init__(self, branch_tokens):
        Branch.__init__(self, branch_tokens)
        self.control_name = branch_tokens[3]


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
        start = -1
        end = -1
        aclist=[]
        acloc = -1
        for li in range(len(lines)):

            if lines[li] =='':
                continue
            else:
                tokens = lines[li].split(" ")
            if tokens[0] in [".circuit",".circuit#"]:
                start = li
            elif tokens[0] in [".end",".end#"]:
                end = li
                if acloc ==-1:
                    acloc = end +1
            elif tokens[0] in [".ac",".ac"]:
                acflag = 1
                print("AC Element")
                acloc = li
                aclist.append((tokens[1],tokens[2]))

        if start > end or start == -1 or end == -1 or acloc<start or acloc<end:
            raise ValueError
        circuit_lines = lines[start+1:end]
        return circuit_lines, aclist
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


def parse(branch,aclist):
    '''Function decides what type of branch is used, 
    and also throws error if branch is invalid'''
    branch_tokens = clean_branch(branch)
    if branch_tokens[0][0] in "RLC":
        parsed_branch = Branch(branch_tokens,aclist)
    elif branch_tokens[0][0] in "VI":
        parsed_branch = Source_Branch(branch_tokens,aclist)
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


def analyse_circuit(circuit_lines, aclines):
    '''Function goes through the circuit line by line and parses it.'''
    list_branches = []
    for branch in circuit_lines:
        if branch_not_comment(branch):
            branch_tokens, parsed_branch = parse(branch, aclines)
            list_branches.append(parsed_branch)
    return list_branches


def generate_nodal_table(circuit_processed_format):
    table = []
    name_list = []
    for ob in circuit_processed_format:
        src = ob.source
        dest = ob.dest
        if src in name_list:
            src_index = name_list.index(src)
        else:
            table.append(Node(src))
            name_list.append(src)
            src_index = name_list.index(src)
        if dest in name_list:
            dest_index = name_list.index(dest)
        else:
            table.append(Node(dest))
            name_list.append(dest)
            dest_index = name_list.index(dest)
        table[src_index].add_branch(ob)
        table[dest_index].add_branch(ob)
    return table , name_list


def construct_matrix(nodal_table, node_name_list, circuit_processed_format,aclist):
    #First we calculate the length of the matrix
    n = len(nodal_table)
    try:
        freq =  float(aclist[0][1])
    except IndexError:
        pass
    IVS = [i for i in circuit_processed_format if i.type == "Independent Voltage Source"]
    k = len(IVS)
    matrix = np.zeros((n+k,n+k),dtype = complex)
    b_values = np.zeros((n+k,1),dtype = complex)
    for i in range(n+k):
        if i==0:
            matrix[0][0] = 1
        else:
            if i<n:
                #in case of the node
                for branch_at_node in (nodal_table[i].from_list+nodal_table[i].to_list):
                    if branch_at_node.type == "Resistor":
                        matrix[i][i]+= 1/branch_at_node.value
                        if branch_at_node.source == node_name_list[i]:
                            matrix[i][node_name_list.index(branch_at_node.dest)] -=1/branch_at_node.value
                        else:
                            matrix[i][node_name_list.index(branch_at_node.source)] -=1/branch_at_node.value
                    elif branch_at_node.type == "Independent Voltage Source":

                        if branch_at_node.source == node_name_list[i]:
                            flag =1
                        else:
                            flag = -1
                        matrix[i][n+IVS.index(branch_at_node)] +=flag
                    elif branch_at_node.type == "Independent Current Source":
                        #current enter is pos
                        if branch_at_node.source == node_name_list[i]:
                            flag =-1
                        else:
                            flag = 1
                        b_values[i] += flag*branch_at_node.value 
                    elif branch_at_node.type == "Capacitor":
                        matrix[i][i]+= (1j)*2*pi*freq*branch_at_node.value
                        if branch_at_node.source == node_name_list[i]:
                            matrix[i][node_name_list.index(branch_at_node.dest)] -=(1j)*2*pi*freq*branch_at_node.value
                        else:
                            matrix[i][node_name_list.index(branch_at_node.source)] -=(1j)*2*pi*freq*branch_at_node.value
                    elif branch_at_node.type == "Inductor":
                        matrix[i][i]+= 1/((1j)*2*pi*freq*branch_at_node.value)
                        if branch_at_node.source == node_name_list[i]:
                            matrix[i][node_name_list.index(branch_at_node.dest)] -=1/((1j)*2*pi*freq*branch_at_node.value)
                        else:
                            matrix[i][node_name_list.index(branch_at_node.source)] -=1/((1j)*2*pi*freq*branch_at_node.value)
            else:
                branch_considered = IVS[i-n]
                matrix[i][node_name_list.index(branch_considered.source)] -=1
                matrix[i][node_name_list.index(branch_considered.dest)] +=1
                if branch_considered.current_type =="ac":
                    phase = branch_considered.phase
                    b_values[i] += branch_considered.value*(math.cos(phase)+(1j)*math.sin(phase))/2
                else:
                    b_values[i] += branch_considered.value
    pprint.pprint(matrix)
    return matrix,b_values,IVS

def main():
    file_name = extract_file_name()
    if file_name is not None:
        circuit_lines, aclist = validate_circuit_existence(file_name)
        circuit_processed_format = analyse_circuit(circuit_lines,aclist)
    nodal_table , node_name_list = generate_nodal_table(circuit_processed_format)
    if "GND" in node_name_list:
        GND_index = node_name_list.index("GND")
        node_name_list = [node_name_list[GND_index]]+node_name_list[:GND_index]+node_name_list[GND_index+1:]
        nodal_table = [nodal_table[GND_index]]+nodal_table[:GND_index]+nodal_table[GND_index+1:]
    matrix,b_values,IVS = construct_matrix(nodal_table,node_name_list,circuit_processed_format,aclist)
    solution = np.linalg.solve(matrix,b_values)
    name_list = ["Voltage at node " +i +" is " for i in node_name_list]+ ["Current through Independent Voltage Source " +i.name+" is " for i in IVS]
    for i in range(len(solution)):
        print(name_list[i],"{:g}".format(solution[i][0]),end="\n")

if __name__ == "__main__":
    main()