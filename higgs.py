#Decision tree assignment 
#The number of leaves determines the VC dimension of your model.
import numpy as np
import unittest, random, math, os, sys
from matplotlib import pyplot as plt

class Node:
    def __init__(self):
        self.data = None
        self.entropy = None
        self.condition = None
        
        self.parent = None
        self.children = None
        

def ID3(d, n, data):
    if not isinstance(data, np.ndarray): data = np.asarray(data)
    #load in data
    #This dataset HAS A HEADER!
    header = data[0]
    data = data[1:]
    root = Node()
    root.data = data
    root.entropy = entropy(data[:,-1])
    root.condition='root'
    root.parent = None
    
    rows, cols = data.shape
    col_data = {}
    for i in range(cols):
        cat, counts = np.unique(data[:,i], return_counts=True)
        decisions = [data[:,-1][data[:,i]==c] for c in cat]
        decisions = [', '.join(d) for d in decisions]
        col_data[header[i]]={}
        col_data[header[i]].update({cat:{'count': c, 'decisions':d} for cat, c, d in zip(cat, counts, decisions)})
        # col_data[header[i]].update({'Total':sum(counts)})


    data_surround = '\n{:{fill}{align}{width}}\n'
    print(data_surround.format('Data Summary', fill='*', align='^', width=50))
    print('n rows : {}, n cols : {}'.format(rows, cols))
    print('Column categories and count:')
    for k, v in col_data.items():
        print('{:>10}: '.format(k))
        for col_cats,val in v.items():
            print('{:>16}: {:>2} : {:>2}'.format(col_cats, val['count'], val['decisions']))
    print(data_surround.format('End Summary', fill='*', align='^', width=50))
    
    buildTree(root, data)

    return root

def buildTree(node, S):
    '''
    buildTree is a helper function for the ID3 and will recursively build a tree.
    The base cases are whether all the indices are the same and therefore cannot be
    split further. In this case, it will return. The tree is built of the initial node, 
    therefore no return value is necessary.
    Parameters
        node (Node) : the node for which children will be spawned
        S (npArray) : the dataset that will be analysed, Labels must be in the last column.
    
    Return
        No return value
    '''
    print('Building tree with dataset...')
    if not isinstance(S, np.ndarray): S = np.asarray(S)  
    if node.entropy is None: node.entropy=entropy(S[:,-1])
    n_features = S.shape[1]-1
    print('Tree is estimated to have {} more levels'.format(n_features))
    print('Calculating the largest gain...')
    gains = [node.entropy - compute_gain(S, i) for i in range(n_features)]
    print(gains)
    max_gain = max(gains)
    print('largest gain is {:0.6f}'.format(max_gain))
    lNode, rNode = Node(), Node()


    return

def load_dataset(path):
    '''
    Load dataset takes in a path to a file and loads it using pandas.

    '''
    if not os.path.isfile(path): return -1
    pass

    

def compute_gain(S, i):
    '''
    Gain computation by splitting the set across the ith index using the entropy calculations
    Parameters:
        S (n-dim array): The dataset that you wish to calculate the information gain on, must 
            be at least 2 dimensions with the labels on the final column.
        i (int) : The index of the column.
    Return:
        gain (float) : The difference between the previous and new entropy 
    '''
    if not isinstance(S, np.ndarray): S = np.asarray(S)  
    rows, cols = S.shape
    if cols < 2: return -1
    if i-1 > cols: return -1
    subset = S[:,[i,-1]]
    rows, cols = subset.shape      
    total_entropy = entropy(subset[:,-1])
    labels = np.unique(subset[:,-1])
    categories = np.unique(subset[:,0])
    divided_S = [subset[subset[:,0]==c] for c in categories]
    entropies = [entropy(div_s[:,-1]) for div_s in divided_S]
    props = [len(div_s)/rows for div_s in divided_S] #count/rows for each category for each column
    combined = sum([x*y for x,y in zip(props, entropies)])
    # print(total_entropy)
    # print(labels)
    # print(categories)
    # print(divided_S)
    # print(entropies)
    # print(props)
    # print(combined)
    return total_entropy - combined

def entropy(S):
    '''
    Calculate the entropy of a dataset across label l
    Parameters:
        S (1-dim array): The dataset that you wish to calculate the entropy on, must be at lest 1 dimension
    Returns:
        entropy (float): The entropy of the column rounded to 6d.p
    '''    
    if not isinstance(S, np.ndarray): S = np.asarray(S)        
    rows = S.shape
    if len(rows) is not 1: return -1
    categories, counts = np.unique(S, return_counts=True)
    cat_cnt = dict(zip(categories, counts))
    
    entropy = -sum((cat_cnt[cat]/rows)*math.log((cat_cnt[cat]/rows), 2) for cat in categories)[0]
    
    return round(entropy, 6)

#Randomly divide the data by the percentage split.
def split_data(data, split):
    print('running split_data')
    largerSplit = split if split > .5 else 1 - split
    test_set = []
    training_set_idx = random.sample(range(len(data)),int(len(data)*largerSplit))
    training_set = [data[i] for i in training_set_idx]
    test_set = [d for d in data if d not in training_set]
    return training_set, test_set

def learning_curve(d, n, training_set, test_set):
    plot = ''
    # you will probably need additional helper functions
    return plot


test_features = ['color', 'softness', 'tasty',] #where tasty is the label
test_data = [
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [1,1,1],
    [1,1,1],
    [1,1,1],
]
common_e = {
    "one_half" : 1.0,
    "one_third" : round(0.9182958340544896, 6),
    'one_quarter' : round(0.8112781244591328, 6),
    "two_fifths": round(0.9709505944546685, 6),
    "one_fifth" : round(0.7219280948873623, 6),
    "one_tenth" : round(0.4689955935892812,6)

}
class TestID3Functions(unittest.TestCase):
    @unittest.skip('Passed')
    def test_entropy(self):
        simpleData = [['a','orange'],['b','apple']]
        data = np.array(simpleData)[:,0]
        e = entropy(data)
        print('entropy for simpleData is {}'.format(e))
        self.assertEqual(e, 1)

    @unittest.skip('Passed')
    def test_another_entropy(self):
        simpleData = [['a','orange'], ['b', 'apple'], ['b', 'apple'],['b','apple']]
        dataLeft = np.array(simpleData)[:,0]
        dataRight = np.array(simpleData)[:,1]
        el = entropy(dataLeft)
        er = entropy(dataRight)
        print('Entropy for simple data L is {}'.format(el))
        print('Entropy for simple data R is {}'.format(er))
        self.assertEqual(el, common_e['one_quarter'])
        self.assertEqual(er, common_e['one_quarter'])

    @unittest.skip('Passed')
    def test_numeric_entropy(self):
        dataColOne = np.array(test_data)[:,0]
        dataColTwo = np.array(test_data)[:,1]
        dataColThree = np.array(test_data)[:,2]
        entropies = [entropy(dataColOne), entropy(dataColTwo), entropy(dataColThree)]
        print('Entropy for test data column 0 is {}'.format(entropies[0]))
        print('Entropy for test data column 1 is {}'.format(entropies[1]))
        print('Entropy for test data column 2(labels) is {}'.format(entropies[2]))
        expected_o = [common_e['two_fifths'], common_e['two_fifths'], round(-((7/10)*math.log(7/10, 2) + (3/10)*math.log(3/10, 2)), 6)]
        self.assertListEqual(entropies, expected_o)

    def test_compute_gain(self):
        print('Running compute Gain Test')

        uncertain_data = np.array([
            [0,'a'],
            [0, 'a'],
            [0, 'b'],
            [1, 'a'],
            [1, 'b']
        ])

        uncertain_to_certain = np.array([
            [0,'a'],
            [0,'a'],
            [0,'a'],
            [1,'b'],
            [1,'b'],
            [1,'b'],
        ])

        uncertain_gain = compute_gain(uncertain_data, 0)
        gain_col0 = compute_gain(test_data,0)
        gain_col1 = compute_gain(test_data,1)
        #root gain - (sum pilog(pi)*pi)
        self.assertEqual(round(uncertain_gain,6), 0.019973)
        self.assertEqual(round(compute_gain(uncertain_to_certain,0),6), 1)

        self.assertEqual(round(gain_col0,6), .556780)
        self.assertEqual(round(gain_col1, 6), .281291)

    @unittest.skip
    def test_split(self):
            print('running test split.')
            data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            split = .3
            train_set, test_set = split_data(data, split)
            # print(train_set)
            # print(test_set)
            self.assertEqual(len(train_set), len(data)*.7)
            self.assertEqual(len(test_set), len(data)*.3)
    
    def test_ID3_level_1_gain(self):
        print('running ID3 init test')
        play_tennis_data = np.array([
            ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Decision'],
            ['Sunny', 'Hot', 'High', 'Weak', 'No'],
            ['Sunny', 'Mild', 'High', 'Weak', 'No'],
            ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
            ['Sunny', 'Cold', 'Normal', 'Weak', 'Yes'],
            ['Sunny', 'Hot', 'High', 'Strong', 'No'],
            ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
            ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
            ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Strong', 'No']]
        )
        #Calculations made using wolframalpha
        root_gain = round(0.96123660472287587, 6)
        expected_gain = {
            'Decision': root_gain,
            'Outlook' : round(root_gain - ((4/13)*0.0 + (4/13)*1.0 + (5/13)*common_e['two_fifths']),6),
            'Temperature': round(root_gain - ((1/13)*0.0 + (2/13)*1.0 + (4/13)*1.0 + (6/13)*common_e['one_third']), 6),
            'Humidity' : round(root_gain - ((7/13)*round(0.98522813603425, 6) + (6/13)*round(0.65002242164835,6))),
            'Wind' : round(root_gain - ((6/13)*1.0 + (7/13)*common_e['one_third']), 6)
        }
        for k, v in expected_gain.items():
            print('{} : {}'.format(k,v))
        tree = ID3('a','b',play_tennis_data)
        self.assertEqual(1, 1)

if __name__ == '__main__':
    #Testing functions
    unittest.main()
    '''
    t = TestingFunctions()
    t_id3 = TestID3Functions()
    t_id3.test_entropy()
    # t.test_split()
    # t_id3.test_ID3()
    '''