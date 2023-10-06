import csv
import numpy as np

with open('MNIST_CSV/mnist_train.csv', 'r') as file:
    csv_reader = csv.reader(file)
    images = []
    count = 0
    for row in csv_reader:
        # Process each row
        count+=1
        label = row[0]
        val = list(np.array(row[1:],dtype=np.float32)/255)
        images.append((label,val))
        f_name ="images/"+ str(label)+"img"+str(count)
        f = open(f_name, "a")
        f.write(str(val))
        f.close()
