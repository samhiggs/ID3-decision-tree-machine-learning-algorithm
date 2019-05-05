#Decision tree assignment 
#The number of leaves determines the VC dimension of your model.
import numpy as np
import unittest, random, math, os, sys
from matplotlib import pyplot as plt

class Node:
    def __init__(self, f, d):
        self.feature = f
        self.decision = d
        self.children = []

    def __str__(self):
       return ''' Node
            {:>7} : {:<4}
            {:>7} : {:<4}
            {:>7} : {:<4}
       '''.format(\
            'Feature', self.feature if self.feature is not None else 'NA',\
            'Decision', self.decision if self.decision is not None else 'NA', \
            'N Children', len(self.children) if self.children else 'False', \
        )

    def __eq__(self, other):
        return(
            self.__class__==other.__class__ and \
            self.feature == other.feature and \
            self.decision == other.decision and \
            self.children == other.children \
        )

data_surround = '\n{:{fill}{align}{width}}\n'


def ID3(d, n, data):
    '''
    ID3 builds a decision tree recursively. Assumes the data has no features
    in the header. Features should be described in the 
    Parameters
        d (int):
            The max depth of the tree 
        n (int):
            The maximum number of nodes
        data (list): 
            n-dimensional dataset
    Returns
        root (Node):
            The root of the tree of depth d.
    '''
    #Check to ensure the inputs are valid.
    if d is None or not isinstance(d, int) or d < 1: raise Exception('d is not valid')
    if n is None or not isinstance(n, int) or n < d or n > len(data[0]): raise Exception('n is not valid')
    if data is None: raise Exception('data is not valid')
    #convert the dataset to a numpy array.
    try:
        if not isinstance(data, np.ndarray): 
            data = np.asarray(data)
        rows, cols = data.shape
        if rows < 2 or cols < 2:
            raise Exception('The dataset will not be useful as there are to few rows and/or columns')
    except Exception as e:
        print(e)
        raise Exception('The data cannot be converted in into a numpy array ')

    features = ['c_'+str(i) for i in range(cols-1)]
    features.append('labels')

    #Setup tree
    root = Node('root', 'root')
    print(data_surround.format('Building Tree', fill='*', align='^', width=50))
    buildTree(data, root, features)
    return root

def buildTree(subset, node, features):
    '''
    buildTree is a helper function for the ID3 and will recursively build a tree.
    The base cases are whether all the indices are the same and therefore cannot be
    split further. In this case, it will return. The tree is built of the initial node, 
    therefore no return value is necessary.
    WARNING: The dataset in the nodes are NOT changed, only the featureset is manipulated.
    This is to avoid excess computation by copying a dataset everytime, instead it just
    points to the one dataset but it's important to use the features as a source of truth.
    Parameters
        node (Node) : 
            the node for which children will be spawned
        target_feature (int) : 
            The index of the target feature relative to the list of features.
        features (list[(String)]):
            List of features remaining in the dataset. Used to label the nodes

    Return
        No return value
    '''
    print(features,subset.shape)
    if not isinstance(subset, np.ndarray): raise Exception('Must be a numpy array')
    if not features or len(features) < 1: raise Exception('No features left.')
    if node is None or node.children is None: raise Exception('No node or improperly created.')
    if subset is None or subset.shape[0] < 1 or subset.shape[1] < 1 or len(features) > subset.shape[1]: 
        raise Exception('subset is not being read in correctly.')
    
    labels = np.unique(subset[:,-1])
    # Base case for if all labels are the same.
    if len(labels) == 1:
        leaf = Node(features[-1], labels[0])
        node.children.append(leaf)
        return

    # Base case for if we are at the end of the dataset
    if len(features) == 1:
        for cat in np.unique(subset[:,-1]):
            leaf = Node(features[0], cat)
            node.children.append(leaf)
        return
    # Make absolutely sure that we don't keep going
    if len(features) == 0: raise Exception('Oops we should not have hit this...Check code!')
   
   #Recursive Function given the best feature of the set (target feature)
    
    max_idx = np.argmax([compute_gain(subset, f)[0] for f, _ in enumerate(features[:-1])])
    feature = features.pop(max_idx)
    for c in np.unique(subset[:, max_idx]):
        #create a child node
        child = Node(feature, c)
        node.children.append(child)
        #split the data
        child_data = subset[subset[:,max_idx]==c]
        child_data = np.concatenate((child_data[:,:max_idx], child_data[:,max_idx+1:]), axis=1)
        buildTree(child_data, child, features)
    return


def visualiseData(data):
    rows, cols = data.shape
    col_data = {}
    for i in range(cols):
        cat, counts = np.unique(data[:,i], return_counts=True)
        decisions = [data[:,-1][data[:,i]==c] for c in cat]
        decisions = [', '.join(d) for d in decisions]
        col_data[i]={}
        col_data[i].update({cat:{'count': c, 'decisions':d} for cat, c, d in zip(cat, counts, decisions)})
        # col_data[header[i]].update({'Total':sum(counts)})

    print(data_surround.format('Data Summary', fill='*', align='^', width=50))
    print('n rows : {}, n cols : {}'.format(rows, cols))
    print('Column categories and count:')
    for k, v in col_data.items():
        print('{:>10}: '.format(k))
        for col_cats,val in v.items():
            print('{:>16}: {:>2} : {:>2}'.format(col_cats, val['count'], val['decisions']))
    print(data_surround.format('End Summary', fill='*', align='^', width=50))

    print(data_surround.format('Visualise The Data', fill='*', align='^', width=50))

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
    categories = np.unique(subset[:,0])
    divided_S = [subset[subset[:,0]==c] for c in categories]
    entropies = [entropy(div_s[:,-1]) for div_s in divided_S]
    props = [len(div_s)/rows for div_s in divided_S] #count/rows for each category for each column
    combined = sum([x*y for x,y in zip(props, entropies)])
    return (total_entropy - combined), categories

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
    
    def test_entropy(self):
        simpleData = [['a','orange'],['b','apple']]
        data = np.array(simpleData)[:,0]
        e = entropy(data)
        print('entropy for simpleData is {}'.format(e))
        self.assertEqual(e, 1)

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
        self.assertEqual(round(uncertain_gain,6), 0.019973)
        self.assertEqual(round(compute_gain(uncertain_to_certain,0),6), 1)
        self.assertEqual(round(gain_col0,6), .556780)
        self.assertEqual(round(gain_col1, 6), .281291)

    def test_split(self):
            print('running test split.')
            data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            split = .3
            train_set, test_set = split_data(data, split)
            # print(train_set)
            # print(test_set)
            self.assertEqual(len(train_set), len(data)*.7)
            self.assertEqual(len(test_set), len(data)*.3)
    
    def test_ID3_temp_data(self):
        play_tennis_data = np.array([
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
        features = ['Outlook', 'Temperature', 'Humidity', \
            'Wind', 'Decision'],

        #Calculations made using wolframalpha
        root_gain = round(0.96123660472287587, 6)
        expected_gain = {
            'Decision': root_gain,
            'Outlook' : round(root_gain - ((4/13)*0.0 + (4/13)*1.0 + (5/13)*common_e['two_fifths']),6),
            'Temperature': round(root_gain - ((1/13)*0.0 + (2/13)*1.0 + (4/13)*1.0 + (6/13)*common_e['one_third']), 6),
            'Humidity' : round(root_gain - ((7/13)*round(0.98522813603425, 6) + (6/13)*round(0.65002242164835,6))),
            'Wind' : round(root_gain - ((6/13)*1.0 + (7/13)*common_e['one_third']), 6)
        }
        root = ID3(3,3,play_tennis_data)
        self.print_tree(root)
        self.assertEqual(1, 1)

    def print_tree(self, root):
        nodes = [[root, root]]
        width, next_width = 1, 0
        depth = 0
        while len(nodes) > 0:
            n, parent = nodes.pop(0)
            if width == 0: 
                depth += 1
                width = next_width
                next_width = 0
            width -= 1

            if len(n.children) > 0:
                next_width+= len(n.children)
                nodes.extend([[child, n] for child in n.children])
            p = depth*2
            print(f"{'':^{p}} Parent {parent.feature} : {parent.decision}")
            print(f"{'':^{p}}{n}")
        return
    def test_tree_build_one_level_perfect_gain(self):
        #Build tree to test.
        data = np.asarray([
            ['sun', 'sun', 'sun', 'cloud', 'cloud'],
            ['go_outside', 'go_outside', 'go_outside', 'stay_indoors', 'stay_indoors']
        ])
        data = data.T
        root = ID3(6,6,data)
        self.assertEqual(1, 1)

    def test_simple_helper(self):
        simple_d = np.array([
            [0,1],
            [0,1],
            [1,0],
            [1,0],
        ])
        node = Node('root', 'root')
        features = ['wind', 'label']
        root = ID3(1,1,simple_d)
        self.print_tree(root)
        self.assertEqual(1,1)

    def test_tree_two_level_imperfect_gain(self):
        test_w_data = np.asarray([
            ['w','C','H',0],
            ['w','C','L',0],
            ['w','C','L',0],
            ['w','H','L',0],
            ['w','H','L',1],
            ['d','H','H',1],
            ['d','H','H',1],
            ['d','H','H',1],
            ['d','C','H',1],
            ['d','C','L',0],
        ])
        
        root = ID3(2,3,test_w_data)
        self.print_tree(root)
        self.assertEqual(1,1)

if __name__ == '__main__':
    #Testing functions
    unittest.main()