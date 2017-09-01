def generate_labelname_label_mapping():
    a = open('train.csv', 'r')
    l = []
    di = {}
    labels = set() 
    for x in a:
        image_name, label =  x.split(',')
        #print image_name, label
        labels.add(label[:-1])
    labels.remove('label')
    i = 0
    for j in labels:
        di[j] = i
        i += 1
        l.append(j)
    return di, l


if __name__ == "__main__":
    print generate_labelname_label_mapping()
