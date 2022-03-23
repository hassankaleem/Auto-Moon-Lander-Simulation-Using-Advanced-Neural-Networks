import csv

with open('C:\\newgame\\ce889assignment-master\\Data Set\\Training Data Set.csv', newline='') as file:
    my_2dlist = []
    my_2dlist = list(csv.reader(file))
    # print(my_2dlist)


def feedforward():
    # print(len(my_2dlist))
    # print(my_2dlist[0][1])
    # print(my_2dlist)

    # my_list  = [1.5,2.5,3.5,5.5]
    # print(min(my_list))

    # print(my_2dlist)
    def normalize_data(column):
        my_new_list = []
        for m in range(len(my_2dlist)):
            my_new_list.append(float(my_2dlist[m][column]))
        # print(my_new_list)
        # print(min((my_new_list)))
        # print(max((my_new_list)))
        # print(type(my_new_list[0]))

        for a in range(len(my_2dlist)):
            my_2dlist[a][column] = (float(my_2dlist[a][column]) - min((my_new_list))) / (
                    max((my_new_list)) - min((my_new_list)))

        # print(min((float(my_new_list)))
        # print(min(my_2dlist[min][column]))

        #
        # for i in range(len(my_2dlist)):
        #     # print(i)
        #     for j in range(len(my_2dlist[i])):
        #         # print(my_2dlist[i][j])
        #         my_2dlist[i][j]= (float(my_2dlist[i][j]) - min((my_new_list)))/(max((my_new_list))-min((my_new_list)))

    normalize_data(0)
    normalize_data(1)
    normalize_data(2)
    normalize_data(3)
    # print(my_2dlist)
    # print(len(my_2dlist))

    # print(my_2dlist(a)(b))


#
#
feedforward()  ########### normailize the data in the main of program #############

########################### 1 STEP OF FEEDFORWARD PROCESS ###################################

############ Input layer weights ###################

a = [[0.5, 0.5],
     [0.5, 0.5],
     [0.5, 0.5]]

########## output layer weights #################
h1 = [[0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5]]

####### save the h1 weights to resue them in layer1 #########
nw_h1 = [[0, 0, 0],
         ########### use to save the initial weights of output weights so that it can be used later for gradient of hidden neurons.
         [0, 0, 0]]

############ save output gradient value #########

out_grad = [[0.0],
            [0.0]]

##########  Target Output ##############################
output = [[0.4994630250176093],
          [0.6025091004814406]]
############ Input #################
b = [[0.5033353915965245],
     [0.3934098608147547]]

############ R_M_S_E #################



R_M_S_error_total = [[],
                     []]

Sum_R_M_s_erro = []




for no_of_epoc in range(2):  ########################## This loop will controll the epoch of the neural network training #################33
    print("****************** This is", no_of_epoc, "epoch*****************************")

    mean_suqare_error = [[],    ## USING This internally for saving square of each error
                         []]
    R_M_S_error_per_epoch = [[0],
                             [0]]

    for ite_rate in range(2):  # add this in parameter >>>>>  len(my_2dlist) #### Iterating to the length of 2dArray #######

        ############## back propogation variable ############

        input_grd_h1 = [[0],  ########### use to save gradient value of h1 , h2 , h3 neurons.
                        [0],
                        [0]]

        input_for_grad = [[0],
                          [0],
                          [0]]

        print(ite_rate)
        import math


        def sigmoid(x):
            return 1 / (1 + math.exp(-x) * (0.9))


        # sigmoid derivatives to adjust synaptic weights
        def sigmoid_derivative(x, e):
            return 0.9 * x * (1 - x) * e


        # sigmoid derivatives to adjust synaptic weights ITS IS FOR hidden neuron to calculate gradient for h1,h2,h3
        def sigmoid_derivative_input_layer(x, al_grad):
            return 0.9 * x * (1 - x) * al_grad


        # list_no = 0

        ############ Input and output assignments from 2d list #################
        b[0][0] = float(my_2dlist[ite_rate][0])  # input
        b[1][0] = float(my_2dlist[ite_rate][1])  # input
        output[0][0] = float(my_2dlist[ite_rate][2])  # output
        output[1][0] = float(my_2dlist[ite_rate][3])  # output
        print("My inputs for layer 1 is 'b'  : ", b)
        print("My expected output is 'output': ", output)

        # if count == 0:
        #     for ab in range(len(my_2dlist)):
        #         for ba in range(len(my_2dlist[ab])):
        #             if ba <=0:  # b[0][0]=my_2dlist[0][0],b[1][0]
        #                 print("initial input",b)
        #                 if ba==0:
        #                     b[ab][0] = my_2dlist[ab][ba]
        #             elif ba==1:
        #                 b[1][0] = my_2dlist[ab][ba]
        #                 print("initial input", b)
        #                 # print(my_2dlist[ab][b])
        #     count+=1

        ############ Hiden Neuron #############
        x = [[0],
             [0],
             [0]]

        #######   output neron #############################
        training_output = [[0],
                           [0]]

        for i in range(len(a)):
            # print('i',i)
            for j in range(len(b[0])):
                # print('j',j)
                for k in range(len(b)):
                    # print('k',k)
                    result = a[i][k] * b[k][j]
                    x[i][j] += result  # weighted sum of input and weights layer 1
                    # print(x[i][j])
            # x[i][j] = sigmoid(result)
            # print(x[i][j])
            # x[i][j] += a[i][k] * b[k][j]

        # for r in range(len(x)):
        #     print(x[r])

        for r in range(len(x)):
            for j in range(len(x[r])):
                # print(sigmoid(x[r][j]))              #sigmoid of input and hidden weights
                x[r][j] = sigmoid(x[r][j])
        print("Weights if output layer is 3x2 matrix 'a'           :", a)
        print("Input of Xvel and Vel is              'b            :", b)
        print("Input/output of hidden layer after sigmoid is 3x1: x=", x)

        for i in range(len(h1)):
            # print('i',i)
            for j in range(len(x[0])):
                # print('j',j)
                for k in range(len(x)):
                    result = h1[i][k] * x[k][j]
                    training_output[i][j] += result  # Weighted sum of hidden layer input and output layer weights

        # print(training_output)
        #
        for r in range(len(training_output)):
            for j in range(len(training_output[r])):
                # print(sigmoid(training_output[r][j]))
                training_output[r][j] = sigmoid(training_output[r][j])  # sigmoid of output layer output

        print('Weights if output layer is 3x2 matrix     :', h1)
        print('Output of Xvel and Yvel is 2x1 Matrix:', training_output)

        # def error(training_output,output):
        #     return training_output - output

        ############################ ERROR CALLCULATION ON OUTPUT LAYER ######################
        error_output = [[0],
                        [0]]

        squre_error = [[0],
                       [0]]





        print('initilize error_output list each time with', error_output)

        # def error_Save():
        #     for i in range(len(error_output)):
        #         for j in range(i):
        #             error_output[i][j] = 1

        for i in range(len(training_output)):

            # print(i)
            # print(training_output[i])
            for j in range(len(output[i])):
                # print(j)
                error_output[i][j] = output[i][j] - training_output[i][j]
                # print("Error at output layer:", training_output[i][j] - output[i][j])

                # # print(output[j])
                # if j==0:
                #     error_output[0][0] = training_output[i][j] - output[i][j]
                # else:
                #     error_output[0][1] = training_output[i][j] - output[i][j]
                #     # print("Error at output layer:", training_output[i][j] - output[i][j])

        # error_Save()
        # print(error_output)
        # error_output[i][j] = training_output[i][j] - output[i][j]
        # print("Error at output layer:", error_output)
        ############################### BACK PROPOGATION Phase ##############################
        # nw_h1 = numpy.array([[0,0,0],
        #          [0,0,0]])

        print('output layer error assignment ', error_output)

        print("Initial error square of error output ", squre_error)
        for val in range(len(squre_error)):
            for j in range(len(squre_error[val])):
                squre_error[val][j] = pow(error_output[val][j],2)
                mean_suqare_error[val].append(squre_error[val][j])

        print("square of error output ",squre_error)
        print("Saving the square of errors in seperate list for each output error. ", mean_suqare_error)



        print('output layer weights before update h1 = ', h1)
        ############################## updating the output layer h1 weights #################
        for val in range(len(h1)):
            # print(h1)
            # print(val)
            for all in range(len(h1[val])):
                gradient = sigmoid_derivative(training_output[val][0],
                                              error_output[val][0])  # lemda * y_val(1-y_val)*error
                out_grad[val][0] = gradient  ########## to be used in next layer
                nw_h1[val][all] = h1[val][all]
                delta_Weight = 0.8 * gradient * x[val][0]  # eta * gradient * hidden sigmoid value ()
                h1[val][all] += delta_Weight
                # print(delta)
                # h1[val][all]=
                # print(error_output[val][0])
                # print(training_output[val][0])
                # print(type(training_output[val]))
                # print(error_output[val])
                # print(all)

        print("old weights to be saved in varible nw_h1", nw_h1)
        print("values of saved gradient on x and y ", out_grad)
        print("updated hidden weights of output layer h1 =", h1)
        print("Initial weights of input layer ", a)

        print('sigmod of hidden layer x= ', x)
        print("Input layer weights", a)

        # for val in range(len(a)):
        #     # print("layer 1 update loop",val)
        #     for all in range(len(a[val])):
        #         # print()
        #         for grad in range(1):
        #             print(val,all,a[val][all],out_grad[grad][0],)
        #             in_gra= sigmoid_derivative_input_layer()
        #         # print(all) # 0:01,1:01,2:01 val>(0,1,2) all(01,01,01)

        # print(nw_h1)
        # print(numpy.transpose(nw_h1))
        # aba = [[1,2,3],
        #        [4,5,6],
        #        [7,8,9]]
        # print(numpy.transpose(aba))

        #

        print("************* transpose of 2 x 3 matric ******************", nw_h1)

        result_transpose = nw_h1
        nw_h1 = [[0, 0],
                 [0, 0],
                 [0, 0]]
        for val in range(len(result_transpose)):
            for j in range(len(result_transpose[val])):
                nw_h1[j][val] = result_transpose[i][j]

        print("************* Result of 2 x 3 matrix to 3 x 2 ******************", nw_h1)

        # print("**",nw_h1)
        #
        # nw_h1 = numpy.transpose(nw_h1)  ######### code the transpose with out numpy ########
        #
        # print("**************",nw_h1)

        print("transpose of hiden weights in output layer nw_h1", nw_h1)
        print("output gradient for x and y out_grad=", out_grad)
        print("initial value for sum of all output layer greadients input_for_grad", input_for_grad)
        # nw_h1=[[1,2],
        #        [3,4],
        #        [5,6]]
        # out_grad=[[1],
        #           [2]]
        for i in range(len(nw_h1)):
            # print('i',i)
            for j in range(len(out_grad[0])):
                # print('j',j)
                for k in range(len(out_grad)):
                    # print('k',k)
                    result = nw_h1[i][k] * out_grad[k][j]
                    input_for_grad[i][j] += result
        print("updated sum of all the gradients and weights ", input_for_grad)

        print("************* Transpose 3 x 2 matrix ******************", nw_h1)

        result_transpose1 = nw_h1
        nw_h1 = [[0, 0, 0],
                 [0, 0, 0]]
        for val in range(len(result_transpose1)):
            for j in range(len(result_transpose1[val])):
                nw_h1[j][val] = result_transpose1[i][j]

        print("*************Result 2 x 3 matrix ******************", nw_h1)

        # print("hello ",nw_h1)
        # nw_h1 = numpy.transpose(nw_h1)  ######### code the transpose with out numpy ########
        # print("hello ", nw_h1)

        print('initial value of hidden neuron gradients gradient h1, h2, h3', input_grd_h1)
        print("sigmoid value of hidden neuron h1 , h2 and h3 is ", x)
        for val in range(len(input_grd_h1)):
            for num in range(len(input_grd_h1[0])):
                print("index is ", val, num, x[val][0], input_for_grad[val][0])
                input_grd_h1[val][num] = sigmoid_derivative_input_layer(x[val][0], input_for_grad[val][0])

        print("saved value of saved gradient h1,h2,h3 = input_grd_h1", input_grd_h1)
        print("input layer weights", a)
        print("input value of x_Vel and y_vel", b)
        ############################## updating the input layer weights #################
        for val in range(len(a)):
            # print(h1)
            # print(val)
            for all in range(len(a[val])):
                print(val, all, input_grd_h1[val][0], b[all][0], a[val][all])  ## b is the input of x vel and y vel
                # for val_of_b in range(1):
                delta_weight = 0.9 * input_grd_h1[val][0] * b[all][0]
                a[val][all] += delta_weight

        print("input weights updated finally ", a)
        # gradient= sigmoid_derivative(training_output[val][0],error_output[val][0])  # lemda * y_val(1-y_val)*error
        # out_grad[val][0]=gradient   ########## to be used in next layer
        # nw_h1[val][all]=h1[val][all]
        # delta_Weight=0.8*gradient*x[val][0]    # eta * gradient * hidden sigmoid value ()
        # h1[val][all] +=delta_Weight

        print("############################### BACK PROPOGATION Phase ##############################")
    print("this my epoch root mean square",R_M_S_error_per_epoch,no_of_epoc)
    for error in range(len(R_M_S_error_per_epoch)):
        for val in range(len(R_M_S_error_per_epoch[error])):
            R_M_S_error_per_epoch[error][val]= math.sqrt(sum(mean_suqare_error[error])/len(mean_suqare_error[error]))
            R_M_S_error_total[error].append(R_M_S_error_per_epoch[error][val])
    print(R_M_S_error_per_epoch)
    print("X AND Y RMSE BUT NOT TOTAL / 2",R_M_S_error_total)
    Sum_R_M_s_erro.append((R_M_S_error_per_epoch[0][0]+R_M_S_error_per_epoch[1][0])/2)
    print("rmse X + Y / 2",Sum_R_M_s_erro)

    ## final weigt of eoch 1 feed for validaiton  ,, 2 epoc of training  same weights of



from matplotlib import pyplot as plt
plt.plot(Sum_R_M_s_erro)
plt.show()




