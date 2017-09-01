a = open('train.csv', 'r')
labels =set() 
for x in a:
    image_name, label =  x.split(',')
    print image_name, label[:-1]
    labels.add(label[:-1])
print len(labels)
