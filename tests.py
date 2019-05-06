import unittest
import numpy as np
import higgs

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

class TestTreeBuild(unittest.TestCase):

    def test_simple_tree(self):
        header = ['decision', 'label']
        data = np.array([
            [0,'y'],
            [1,'y'],
        ])
        d = 1
        n = 3
        ID3 = higgs.ID3(d,n,data,header)
        ID3.fit()
        root = ID3.root
        print(root)
        self.assertEqual(root.feature, 'decision')
        self.assertEqual(root.children[0].feature, 'label')
        self.assertEqual(root.children[0].decision, 'y')
        self.assertEqual(root.children[1].feature, 'label')
        self.assertEqual(root.children[1].decision, 'n')


class TestID3Functions(unittest.TestCase):
    def test_entropy(self):
        simpleData = [['a','orange'],['b','apple']]
        data = np.array(simpleData)[:,0]
        # e = higgs.ID3.entropy(data)
        # print('entropy for simpleData is {}'.format(e))
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
    @unittest.skip('Not Ready')
    def test_d_and_n(self):
        #Import data
        data_fn = 'data\house-votes-84.data'
        names = [
            'Class Name',
            'handicapped-infants',
            'water-project-cost-sharing',
            'adoption-of-the-budget-resolution',
            'physician-fee-freeze',
            'el-salvador-aid',
            'religious-groups-in-schools',
            'anti-satellite-test-ban',
            'aid-to-nicaraguan-contras',
            'mx-missile',
            'immigration',
            'synfuels-corporation-cutback',
            'education-spending',
            'superfund-right-to-sue',
            'crime',
            'duty-free-exports',
            'export-administration-act-south-africa',
        ]
        names = [ name.replace(' ', '_').lower() for name in names]
        data = pd.read_csv(data_fn,names=names )
        data = data.replace('?', np.nan)
        data = data.fillna(method='pad')
        data = data.fillna('y')
        n = names.pop(0)
        names.append(n)
        data = data[names]
        root = ID3(3,3,data.values, names)
        self.print_tree(root)
        self.assertEqual(1,1)
    @unittest.skip('Not Ready')
    def test_predict(self):
            data_fn = 'data\house-votes-84.data'
            names = [
                'Class Name',
                'handicapped-infants',
                'water-project-cost-sharing',
                'adoption-of-the-budget-resolution',
                'physician-fee-freeze',
                'el-salvador-aid',
                'religious-groups-in-schools',
                'anti-satellite-test-ban',
                'aid-to-nicaraguan-contras',
                'mx-missile',
                'immigration',
                'synfuels-corporation-cutback',
                'education-spending',
                'superfund-right-to-sue',
                'crime',
                'duty-free-exports',
                'export-administration-act-south-africa',
            ]
            names = [ name.replace(' ', '_').lower() for name in names]
            data = pd.read_csv(data_fn,names=names )
            data = data.replace('?', np.nan)
            data = data.fillna(method='pad')
            data = data.fillna('y')
            n = names.pop(0)
            names.append(n)
            data = data[names]
            train, test = split_data(data.values, .8)
            root = learning_curve(6,40,train, test, names)
            self.assertEqual(1,1)

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
                print(f"{' ':^{p}} Parent {parent.feature} : {parent.decision}")
                print(f"{' ':^{p+2}} {n.feature} : {n.decision} : {len(n.children)}")
            return
