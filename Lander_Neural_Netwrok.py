import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x)*(0.9))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x,e):
    return 0.9 * x * (1 - x)*e
##########  Target Output ##############################
output = [[0.4994630250176093],
          [0.6025091004814406]]
############ Input layer weights ###################
a=[[0.5,0.5],
   [0.5,0.5],
   [0.5,0.5]]
############ Input #################
b=[[0.5033353915965245],
   [0.3934098608147547]]
############ Hiden Neuron #############
x=[[0],
   [0],
   [0]]
########## output layer weights #################
h1=[[0.5,0.5,0.5],
   [0.5,0.5,0.5]]
#######   output neron #############################
training_output=[[0],
                [0]]



for i in range(len(a)):
    # print('i',i)
    for j in range(len(b[0])):
        # print('j',j)
        for k in range(len(b)):
            # print('k',k)
            result = a[i][k] * b[k][j]
            x[i][j] += result                   #weighted sum of input and weights layer 1
            # print(x[i][j])
    # x[i][j] = sigmoid(result)
    # print(x[i][j])
            # x[i][j] += a[i][k] * b[k][j]


# for r in range(len(x)):
#     print(x[r])

for r in range(len(x)):
    for j in range(len(x[r])):
        # print(sigmoid(x[r][j]))              #sigmoid of input and hidden weights
        x[r][j]= sigmoid(x[r][j])
print('Weights if output layer is 3x2 matrix            :', a)
print('Input of Xvel and Vel is                         :',b)
print('Input/output of hidden layer after sigmoid is 3x1:',x)

for i in range(len(h1)):
    # print('i',i)
    for j in range(len(x[0])):
        # print('j',j)
        for k in range(len(x)):
            result = h1[i][k] * x[k][j]
            training_output[i][j] += result # Weighted sum of hidden layer input and output layer weights

# print(training_output)
#
for r in range(len(training_output)):
    for j in range(len(training_output[r])):
        # print(sigmoid(training_output[r][j]))
        training_output[r][j] = sigmoid(training_output[r][j]) #sigmoid of output layer output

print('Weights if output layer is 3x2 matrix     :', h1)
print('Output of Xvel and Yvel is 2x1 Matrix:', training_output)

# def error(training_output,output):
#     return training_output - output

        ############################ ERROR CALLCULATION ON OUTPUT LAYER ######################
error_output= [[0],
               [0]]
print(error_output)

def error_Save():
    for i in range(len(error_output)):
        for j in range(i):
            error_output[i][j]=1



for i in range(len(training_output)):
    # print(training_output[i])
    for j in range(len(output[i])):
        # print(output[j])
        print("Error at output layer:",training_output[i][j] - output[i][j])
error_Save()
print(error_output)
        # error_output[i][j] = training_output[i][j] - output[i][j]
        # print("Error at output layer:", error_output)
        ############################### BACK PROPOGATION Phase ##############################