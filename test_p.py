# test_print.py
root = {}
root['info'] = ['root', 1.0]
root['children'] = [['Temperature', .9235], ['Humidity', .67865]]
print('{:^85}\n{:^85}'.format(*root['info']))
alignment = 40
node_x =2
for i in range(3):
    node_x+=2*i
    alignment -= 2*i
    print('{:^{}}{:^{}} {:^{}} '.format('',alignment, '/', node_x, '\\', node_x))
print('{:^{}}{:^{}} {:^{}}'.format('', alignment-6, root['children'][0][0], node_x-2, root['children'][1][0], node_x+6))