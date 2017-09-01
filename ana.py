if __name__ == '__main__':
    main()
l = []
dict = {}
def main():
    a = open('train.csv', 'r')
    labels =set() 
    for x in a:
        image_name, label =  x.split(',')
        #print image_name, label
        labels.add(label[:-1])
    labels.remove('label')
    i = 0
    for j in labels:
        dict[j] = i
        i += 1
        l.append(j)
def returndict():       #returns dictionary
    return dict
def returnlist():       #returns list
    return l

"""
import ana
ana.main()
list = ana.returnlist()
dict = ana.returndict()
"""
