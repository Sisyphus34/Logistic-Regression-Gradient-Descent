

import numpy as np

def create_features():
    rows = 0;
    features = 0;
    
    filename = input("Please enter filename including extension for training data: \n")

    data_file = open(filename, "r")
    
    first_line = data_file.readline().strip()
        
    data_params = first_line.split('\t')
    
    rows = int(data_params[0])
    features = int(data_params[1])
    prediction = 1
    
    print("Rows: ", rows)
    print("Features: ", features)
    
    x_data = np.zeros([rows, (features + prediction)])
    y_data = np.zeros([rows, 1])
    x_data_nosub0 = np.zeros([rows, features])

    last_col = features
    # print("lines: ", rows, " features: ", features)
    fout1 = open("new_train.txt", 'w')
    fout1.write((str(rows) + "\t" + str(8) + "\n"))
    for i in range(rows):
        row = data_file.readline().strip()
        x = row.split("\t")
        # print("x = ", x)

        # for j in range(features + prediction):
        #     if j == last_col:
        #         y_data[i, 0] = float(x[j])
        #     else:
        #         x_data_nosub0[i, j] = float(x[j])

        
        thePower = 2
        print(x)
        for j in range(thePower+1):
            for c in range(thePower+1):
                temp = (float(x[0])**c)*(float(x[1])**j)
                print("temp = ", temp)
                if (temp != 1):
                    fout1.write(str(temp)+"\t")
        fout1.write(str(x[2])+"\n")
       

    data_file.close()
    fout1.close()
    # print("x_data: ", x_data_nosub0)
    # print("y_data: ", y_data)


    # fout = open("added_features.txt", "w")
    # thePower = 2
    # count = 0
    # for b in range(thePower+1):
    #     for a in range(thePower+1):
    #         for c in x_data_nosub0:
    #             print("c = ", c)
    #             temp = (c[0]**a)*(c[1]**b)
    #             print("temp = ", temp, a, b)
    #             if (temp != 1):
    #                 fout.write(str(temp)+"\t")
    #     fout.write(str(y_data[count])+"\n")
    #     count += 1
create_features()
