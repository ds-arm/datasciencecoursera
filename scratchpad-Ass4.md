```python
import pip
for m in pip.get_installed_distributions():
    if m.project_name in ["gensim", "nltk", "numpy", "pandas", "pickleshare", "scipy", "scikit-learn", "networkx"]:
        print(m.project_name, ":", m.version)
```

    scipy : 0.19.0
    scikit-learn : 0.18.1
    pickleshare : 0.7.4
    pandas : 0.20.1
    numpy : 1.12.1
    nltk : 3.2.4
    networkx : 1.11
    gensim : 3.4.0
    


```python
import sys
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mpl
import numpy as np
import scipy as sp
import IPython
import sklearn as skl
import networkx as nx
```


```python
print('Python version is: {0}'.format(sys.version))
print('Pandas version is: {0}'.format(pd.__version__))
print('Matplotlib version is: {0}'.format(mpl.__version__))
print('Numpy version is: {0}'.format(np.__version__))
print('Scipy version is: {0}'.format(sp.__version__))
print('IPython version is: {0}'.format(IPython.__version__))
print('Scikitlearn version is: {0}'.format(skl.__version__))
print('NetworkX version is: {0}'.format(nx.__version__))
```

    Python version is: 3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:25:24) [MSC v.1900 64 bit (AMD64)]
    Pandas version is: 0.20.1
    Matplotlib version is: 2.0.2
    Numpy version is: 1.12.1
    Scipy version is: 0.19.0
    IPython version is: 5.3.0
    Scikitlearn version is: 0.18.1
    NetworkX version is: 1.11
    


```python
# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    %matplotlib notebook
    import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);
```


```python
import pickle
import matplotlib.pyplot as plt
%matplotlib notebook
```

# Part 2

## Part 2A - Salary prediction


```python
G = nx.read_gpickle('email_prediction.txt')
print(nx.info(G))
```

    Name: 
    Type: Graph
    Number of nodes: 1005
    Number of edges: 16706
    Average degree:  33.2458
    


```python
nx.is_connected(G)
```




    False




```python
nx.number_connected_components(G)
```




    20




```python
G.nodes(data=True)
```




    [(0, {'Department': 1, 'ManagementSalary': 0.0}),
     (1, {'Department': 1, 'ManagementSalary': nan}),
     (2, {'Department': 21, 'ManagementSalary': nan}),
     (3, {'Department': 21, 'ManagementSalary': 1.0}),
     (4, {'Department': 21, 'ManagementSalary': 1.0}),
     (5, {'Department': 25, 'ManagementSalary': nan}),
     (6, {'Department': 25, 'ManagementSalary': 1.0}),
     (7, {'Department': 14, 'ManagementSalary': 0.0}),
     (8, {'Department': 14, 'ManagementSalary': nan}),
     (9, {'Department': 14, 'ManagementSalary': 0.0}),
     (10, {'Department': 9, 'ManagementSalary': 0.0}),
     (11, {'Department': 14, 'ManagementSalary': 0.0}),
     (12, {'Department': 14, 'ManagementSalary': 1.0}),
     (13, {'Department': 26, 'ManagementSalary': 1.0}),
     (14, {'Department': 4, 'ManagementSalary': nan}),
     (15, {'Department': 17, 'ManagementSalary': 0.0}),
     (16, {'Department': 34, 'ManagementSalary': 0.0}),
     (17, {'Department': 1, 'ManagementSalary': 0.0}),
     (18, {'Department': 1, 'ManagementSalary': nan}),
     (19, {'Department': 14, 'ManagementSalary': 0.0}),
     (20, {'Department': 9, 'ManagementSalary': 0.0}),
     (21, {'Department': 9, 'ManagementSalary': 1.0}),
     (22, {'Department': 9, 'ManagementSalary': 0.0}),
     (23, {'Department': 11, 'ManagementSalary': 0.0}),
     (24, {'Department': 11, 'ManagementSalary': 0.0}),
     (25, {'Department': 11, 'ManagementSalary': 0.0}),
     (26, {'Department': 11, 'ManagementSalary': 0.0}),
     (27, {'Department': 11, 'ManagementSalary': nan}),
     (28, {'Department': 11, 'ManagementSalary': 1.0}),
     (29, {'Department': 11, 'ManagementSalary': 1.0}),
     (30, {'Department': 11, 'ManagementSalary': nan}),
     (31, {'Department': 11, 'ManagementSalary': nan}),
     (32, {'Department': 11, 'ManagementSalary': 0.0}),
     (33, {'Department': 11, 'ManagementSalary': 0.0}),
     (34, {'Department': 11, 'ManagementSalary': nan}),
     (35, {'Department': 11, 'ManagementSalary': 1.0}),
     (36, {'Department': 11, 'ManagementSalary': 1.0}),
     (37, {'Department': 11, 'ManagementSalary': nan}),
     (38, {'Department': 11, 'ManagementSalary': 0.0}),
     (39, {'Department': 11, 'ManagementSalary': 0.0}),
     (40, {'Department': 11, 'ManagementSalary': nan}),
     (41, {'Department': 5, 'ManagementSalary': 0.0}),
     (42, {'Department': 34, 'ManagementSalary': 0.0}),
     (43, {'Department': 14, 'ManagementSalary': 0.0}),
     (44, {'Department': 14, 'ManagementSalary': 1.0}),
     (45, {'Department': 17, 'ManagementSalary': nan}),
     (46, {'Department': 17, 'ManagementSalary': 0.0}),
     (47, {'Department': 10, 'ManagementSalary': 1.0}),
     (48, {'Department': 10, 'ManagementSalary': 0.0}),
     (49, {'Department': 36, 'ManagementSalary': 0.0}),
     (50, {'Department': 37, 'ManagementSalary': 0.0}),
     (51, {'Department': 5, 'ManagementSalary': 0.0}),
     (52, {'Department': 7, 'ManagementSalary': 0.0}),
     (53, {'Department': 4, 'ManagementSalary': 0.0}),
     (54, {'Department': 22, 'ManagementSalary': nan}),
     (55, {'Department': 22, 'ManagementSalary': nan}),
     (56, {'Department': 21, 'ManagementSalary': 0.0}),
     (57, {'Department': 21, 'ManagementSalary': 1.0}),
     (58, {'Department': 21, 'ManagementSalary': 1.0}),
     (59, {'Department': 21, 'ManagementSalary': 0.0}),
     (60, {'Department': 7, 'ManagementSalary': nan}),
     (61, {'Department': 7, 'ManagementSalary': 0.0}),
     (62, {'Department': 36, 'ManagementSalary': nan}),
     (63, {'Department': 21, 'ManagementSalary': 1.0}),
     (64, {'Department': 25, 'ManagementSalary': 1.0}),
     (65, {'Department': 4, 'ManagementSalary': nan}),
     (66, {'Department': 8, 'ManagementSalary': 0.0}),
     (67, {'Department': 15, 'ManagementSalary': 0.0}),
     (68, {'Department': 15, 'ManagementSalary': 0.0}),
     (69, {'Department': 15, 'ManagementSalary': 0.0}),
     (70, {'Department': 37, 'ManagementSalary': 0.0}),
     (71, {'Department': 37, 'ManagementSalary': 0.0}),
     (72, {'Department': 9, 'ManagementSalary': 0.0}),
     (73, {'Department': 1, 'ManagementSalary': 0.0}),
     (74, {'Department': 1, 'ManagementSalary': 0.0}),
     (75, {'Department': 10, 'ManagementSalary': 0.0}),
     (76, {'Department': 10, 'ManagementSalary': 0.0}),
     (77, {'Department': 3, 'ManagementSalary': nan}),
     (78, {'Department': 3, 'ManagementSalary': 0.0}),
     (79, {'Department': 3, 'ManagementSalary': nan}),
     (80, {'Department': 29, 'ManagementSalary': 0.0}),
     (81, {'Department': 15, 'ManagementSalary': 1.0}),
     (82, {'Department': 36, 'ManagementSalary': 1.0}),
     (83, {'Department': 36, 'ManagementSalary': 1.0}),
     (84, {'Department': 37, 'ManagementSalary': 1.0}),
     (85, {'Department': 1, 'ManagementSalary': 0.0}),
     (86, {'Department': 36, 'ManagementSalary': 1.0}),
     (87, {'Department': 34, 'ManagementSalary': 1.0}),
     (88, {'Department': 20, 'ManagementSalary': 0.0}),
     (89, {'Department': 20, 'ManagementSalary': 0.0}),
     (90, {'Department': 8, 'ManagementSalary': 0.0}),
     (91, {'Department': 15, 'ManagementSalary': 0.0}),
     (92, {'Department': 9, 'ManagementSalary': 0.0}),
     (93, {'Department': 4, 'ManagementSalary': 0.0}),
     (94, {'Department': 5, 'ManagementSalary': 0.0}),
     (95, {'Department': 4, 'ManagementSalary': 0.0}),
     (96, {'Department': 20, 'ManagementSalary': 1.0}),
     (97, {'Department': 16, 'ManagementSalary': nan}),
     (98, {'Department': 16, 'ManagementSalary': 0.0}),
     (99, {'Department': 16, 'ManagementSalary': 0.0}),
     (100, {'Department': 16, 'ManagementSalary': 0.0}),
     (101, {'Department': 16, 'ManagementSalary': nan}),
     (102, {'Department': 38, 'ManagementSalary': 0.0}),
     (103, {'Department': 7, 'ManagementSalary': nan}),
     (104, {'Department': 7, 'ManagementSalary': 0.0}),
     (105, {'Department': 34, 'ManagementSalary': 1.0}),
     (106, {'Department': 38, 'ManagementSalary': 1.0}),
     (107, {'Department': 36, 'ManagementSalary': 1.0}),
     (108, {'Department': 8, 'ManagementSalary': nan}),
     (109, {'Department': 27, 'ManagementSalary': 0.0}),
     (110, {'Department': 8, 'ManagementSalary': 0.0}),
     (111, {'Department': 8, 'ManagementSalary': 0.0}),
     (112, {'Department': 8, 'ManagementSalary': 0.0}),
     (113, {'Department': 10, 'ManagementSalary': nan}),
     (114, {'Department': 10, 'ManagementSalary': 1.0}),
     (115, {'Department': 13, 'ManagementSalary': 1.0}),
     (116, {'Department': 13, 'ManagementSalary': 0.0}),
     (117, {'Department': 6, 'ManagementSalary': 0.0}),
     (118, {'Department': 26, 'ManagementSalary': 0.0}),
     (119, {'Department': 10, 'ManagementSalary': 0.0}),
     (120, {'Department': 1, 'ManagementSalary': 0.0}),
     (121, {'Department': 36, 'ManagementSalary': 1.0}),
     (122, {'Department': 0, 'ManagementSalary': nan}),
     (123, {'Department': 13, 'ManagementSalary': 0.0}),
     (124, {'Department': 16, 'ManagementSalary': 0.0}),
     (125, {'Department': 16, 'ManagementSalary': 0.0}),
     (126, {'Department': 22, 'ManagementSalary': 0.0}),
     (127, {'Department': 6, 'ManagementSalary': 0.0}),
     (128, {'Department': 5, 'ManagementSalary': 1.0}),
     (129, {'Department': 4, 'ManagementSalary': 1.0}),
     (130, {'Department': 0, 'ManagementSalary': 0.0}),
     (131, {'Department': 28, 'ManagementSalary': 1.0}),
     (132, {'Department': 28, 'ManagementSalary': 1.0}),
     (133, {'Department': 4, 'ManagementSalary': 1.0}),
     (134, {'Department': 2, 'ManagementSalary': 0.0}),
     (135, {'Department': 13, 'ManagementSalary': 1.0}),
     (136, {'Department': 13, 'ManagementSalary': 1.0}),
     (137, {'Department': 21, 'ManagementSalary': 1.0}),
     (138, {'Department': 21, 'ManagementSalary': 1.0}),
     (139, {'Department': 17, 'ManagementSalary': 0.0}),
     (140, {'Department': 17, 'ManagementSalary': 0.0}),
     (141, {'Department': 14, 'ManagementSalary': nan}),
     (142, {'Department': 36, 'ManagementSalary': nan}),
     (143, {'Department': 8, 'ManagementSalary': 0.0}),
     (144, {'Department': 40, 'ManagementSalary': nan}),
     (145, {'Department': 35, 'ManagementSalary': nan}),
     (146, {'Department': 15, 'ManagementSalary': 0.0}),
     (147, {'Department': 23, 'ManagementSalary': 1.0}),
     (148, {'Department': 0, 'ManagementSalary': 0.0}),
     (149, {'Department': 0, 'ManagementSalary': 0.0}),
     (150, {'Department': 7, 'ManagementSalary': nan}),
     (151, {'Department': 10, 'ManagementSalary': 1.0}),
     (152, {'Department': 37, 'ManagementSalary': 0.0}),
     (153, {'Department': 27, 'ManagementSalary': 1.0}),
     (154, {'Department': 35, 'ManagementSalary': nan}),
     (155, {'Department': 35, 'ManagementSalary': 0.0}),
     (156, {'Department': 0, 'ManagementSalary': 0.0}),
     (157, {'Department': 0, 'ManagementSalary': 0.0}),
     (158, {'Department': 19, 'ManagementSalary': nan}),
     (159, {'Department': 19, 'ManagementSalary': 0.0}),
     (160, {'Department': 36, 'ManagementSalary': 1.0}),
     (161, {'Department': 14, 'ManagementSalary': 0.0}),
     (162, {'Department': 37, 'ManagementSalary': 0.0}),
     (163, {'Department': 24, 'ManagementSalary': 0.0}),
     (164, {'Department': 17, 'ManagementSalary': 0.0}),
     (165, {'Department': 13, 'ManagementSalary': 1.0}),
     (166, {'Department': 36, 'ManagementSalary': 1.0}),
     (167, {'Department': 4, 'ManagementSalary': 0.0}),
     (168, {'Department': 4, 'ManagementSalary': 0.0}),
     (169, {'Department': 13, 'ManagementSalary': 0.0}),
     (170, {'Department': 13, 'ManagementSalary': 1.0}),
     (171, {'Department': 10, 'ManagementSalary': 1.0}),
     (172, {'Department': 4, 'ManagementSalary': 0.0}),
     (173, {'Department': 38, 'ManagementSalary': 0.0}),
     (174, {'Department': 32, 'ManagementSalary': 0.0}),
     (175, {'Department': 32, 'ManagementSalary': nan}),
     (176, {'Department': 4, 'ManagementSalary': 0.0}),
     (177, {'Department': 1, 'ManagementSalary': 0.0}),
     (178, {'Department': 0, 'ManagementSalary': 0.0}),
     (179, {'Department': 0, 'ManagementSalary': 0.0}),
     (180, {'Department': 0, 'ManagementSalary': 0.0}),
     (181, {'Department': 7, 'ManagementSalary': nan}),
     (182, {'Department': 7, 'ManagementSalary': 0.0}),
     (183, {'Department': 4, 'ManagementSalary': 1.0}),
     (184, {'Department': 15, 'ManagementSalary': 0.0}),
     (185, {'Department': 16, 'ManagementSalary': 0.0}),
     (186, {'Department': 40, 'ManagementSalary': 0.0}),
     (187, {'Department': 15, 'ManagementSalary': 1.0}),
     (188, {'Department': 15, 'ManagementSalary': 0.0}),
     (189, {'Department': 15, 'ManagementSalary': 1.0}),
     (190, {'Department': 15, 'ManagementSalary': 0.0}),
     (191, {'Department': 0, 'ManagementSalary': 1.0}),
     (192, {'Department': 21, 'ManagementSalary': 0.0}),
     (193, {'Department': 21, 'ManagementSalary': nan}),
     (194, {'Department': 21, 'ManagementSalary': 0.0}),
     (195, {'Department': 21, 'ManagementSalary': 0.0}),
     (196, {'Department': 5, 'ManagementSalary': nan}),
     (197, {'Department': 4, 'ManagementSalary': 1.0}),
     (198, {'Department': 4, 'ManagementSalary': 1.0}),
     (199, {'Department': 4, 'ManagementSalary': 0.0}),
     (200, {'Department': 4, 'ManagementSalary': nan}),
     (201, {'Department': 4, 'ManagementSalary': 1.0}),
     (202, {'Department': 4, 'ManagementSalary': nan}),
     (203, {'Department': 4, 'ManagementSalary': 0.0}),
     (204, {'Department': 5, 'ManagementSalary': nan}),
     (205, {'Department': 5, 'ManagementSalary': 0.0}),
     (206, {'Department': 4, 'ManagementSalary': 0.0}),
     (207, {'Department': 4, 'ManagementSalary': 0.0}),
     (208, {'Department': 22, 'ManagementSalary': 0.0}),
     (209, {'Department': 19, 'ManagementSalary': 1.0}),
     (210, {'Department': 19, 'ManagementSalary': 1.0}),
     (211, {'Department': 22, 'ManagementSalary': 1.0}),
     (212, {'Department': 34, 'ManagementSalary': 0.0}),
     (213, {'Department': 14, 'ManagementSalary': 0.0}),
     (214, {'Department': 0, 'ManagementSalary': nan}),
     (215, {'Department': 1, 'ManagementSalary': nan}),
     (216, {'Department': 17, 'ManagementSalary': 0.0}),
     (217, {'Department': 37, 'ManagementSalary': 0.0}),
     (218, {'Department': 1, 'ManagementSalary': 0.0}),
     (219, {'Department': 1, 'ManagementSalary': 0.0}),
     (220, {'Department': 1, 'ManagementSalary': 0.0}),
     (221, {'Department': 1, 'ManagementSalary': 0.0}),
     (222, {'Department': 1, 'ManagementSalary': 0.0}),
     (223, {'Department': 1, 'ManagementSalary': 0.0}),
     (224, {'Department': 1, 'ManagementSalary': 0.0}),
     (225, {'Department': 1, 'ManagementSalary': 0.0}),
     (226, {'Department': 1, 'ManagementSalary': 0.0}),
     (227, {'Department': 1, 'ManagementSalary': 0.0}),
     (228, {'Department': 1, 'ManagementSalary': 0.0}),
     (229, {'Department': 10, 'ManagementSalary': 0.0}),
     (230, {'Department': 23, 'ManagementSalary': nan}),
     (231, {'Department': 0, 'ManagementSalary': nan}),
     (232, {'Department': 4, 'ManagementSalary': 1.0}),
     (233, {'Department': 19, 'ManagementSalary': 0.0}),
     (234, {'Department': 19, 'ManagementSalary': 0.0}),
     (235, {'Department': 19, 'ManagementSalary': 0.0}),
     (236, {'Department': 19, 'ManagementSalary': nan}),
     (237, {'Department': 19, 'ManagementSalary': 0.0}),
     (238, {'Department': 19, 'ManagementSalary': 0.0}),
     (239, {'Department': 19, 'ManagementSalary': nan}),
     (240, {'Department': 19, 'ManagementSalary': nan}),
     (241, {'Department': 19, 'ManagementSalary': 0.0}),
     (242, {'Department': 19, 'ManagementSalary': 0.0}),
     (243, {'Department': 19, 'ManagementSalary': 0.0}),
     (244, {'Department': 19, 'ManagementSalary': 0.0}),
     (245, {'Department': 10, 'ManagementSalary': 0.0}),
     (246, {'Department': 14, 'ManagementSalary': 0.0}),
     (247, {'Department': 14, 'ManagementSalary': nan}),
     (248, {'Department': 1, 'ManagementSalary': 0.0}),
     (249, {'Department': 14, 'ManagementSalary': nan}),
     (250, {'Department': 7, 'ManagementSalary': 0.0}),
     (251, {'Department': 13, 'ManagementSalary': nan}),
     (252, {'Department': 20, 'ManagementSalary': nan}),
     (253, {'Department': 31, 'ManagementSalary': nan}),
     (254, {'Department': 40, 'ManagementSalary': 0.0}),
     (255, {'Department': 6, 'ManagementSalary': nan}),
     (256, {'Department': 4, 'ManagementSalary': 0.0}),
     (257, {'Department': 0, 'ManagementSalary': 0.0}),
     (258, {'Department': 8, 'ManagementSalary': 0.0}),
     (259, {'Department': 9, 'ManagementSalary': 0.0}),
     (260, {'Department': 9, 'ManagementSalary': 0.0}),
     (261, {'Department': 10, 'ManagementSalary': nan}),
     (262, {'Department': 0, 'ManagementSalary': nan}),
     (263, {'Department': 10, 'ManagementSalary': 0.0}),
     (264, {'Department': 14, 'ManagementSalary': nan}),
     (265, {'Department': 14, 'ManagementSalary': 0.0}),
     (266, {'Department': 14, 'ManagementSalary': nan}),
     (267, {'Department': 14, 'ManagementSalary': nan}),
     (268, {'Department': 39, 'ManagementSalary': 0.0}),
     (269, {'Department': 17, 'ManagementSalary': 1.0}),
     (270, {'Department': 4, 'ManagementSalary': nan}),
     (271, {'Department': 28, 'ManagementSalary': nan}),
     (272, {'Department': 17, 'ManagementSalary': nan}),
     (273, {'Department': 17, 'ManagementSalary': 0.0}),
     (274, {'Department': 17, 'ManagementSalary': 0.0}),
     (275, {'Department': 4, 'ManagementSalary': 0.0}),
     (276, {'Department': 4, 'ManagementSalary': 0.0}),
     (277, {'Department': 0, 'ManagementSalary': 0.0}),
     (278, {'Department': 0, 'ManagementSalary': nan}),
     (279, {'Department': 23, 'ManagementSalary': nan}),
     (280, {'Department': 4, 'ManagementSalary': 1.0}),
     (281, {'Department': 21, 'ManagementSalary': 1.0}),
     (282, {'Department': 36, 'ManagementSalary': 1.0}),
     (283, {'Department': 36, 'ManagementSalary': nan}),
     (284, {'Department': 0, 'ManagementSalary': 0.0}),
     (285, {'Department': 22, 'ManagementSalary': 1.0}),
     (286, {'Department': 21, 'ManagementSalary': 0.0}),
     (287, {'Department': 15, 'ManagementSalary': 0.0}),
     (288, {'Department': 37, 'ManagementSalary': 0.0}),
     (289, {'Department': 0, 'ManagementSalary': 0.0}),
     (290, {'Department': 4, 'ManagementSalary': 1.0}),
     (291, {'Department': 4, 'ManagementSalary': 0.0}),
     (292, {'Department': 4, 'ManagementSalary': 1.0}),
     (293, {'Department': 14, 'ManagementSalary': 0.0}),
     (294, {'Department': 4, 'ManagementSalary': nan}),
     (295, {'Department': 7, 'ManagementSalary': nan}),
     (296, {'Department': 7, 'ManagementSalary': 0.0}),
     (297, {'Department': 1, 'ManagementSalary': 0.0}),
     (298, {'Department': 15, 'ManagementSalary': nan}),
     (299, {'Department': 15, 'ManagementSalary': nan}),
     (300, {'Department': 38, 'ManagementSalary': 0.0}),
     (301, {'Department': 26, 'ManagementSalary': 1.0}),
     (302, {'Department': 20, 'ManagementSalary': 0.0}),
     (303, {'Department': 20, 'ManagementSalary': 0.0}),
     (304, {'Department': 20, 'ManagementSalary': 0.0}),
     (305, {'Department': 21, 'ManagementSalary': 0.0}),
     (306, {'Department': 9, 'ManagementSalary': nan}),
     (307, {'Department': 1, 'ManagementSalary': 0.0}),
     (308, {'Department': 1, 'ManagementSalary': nan}),
     (309, {'Department': 1, 'ManagementSalary': 0.0}),
     (310, {'Department': 1, 'ManagementSalary': nan}),
     (311, {'Department': 1, 'ManagementSalary': nan}),
     (312, {'Department': 1, 'ManagementSalary': nan}),
     (313, {'Department': 1, 'ManagementSalary': 0.0}),
     (314, {'Department': 1, 'ManagementSalary': 0.0}),
     (315, {'Department': 1, 'ManagementSalary': nan}),
     (316, {'Department': 1, 'ManagementSalary': nan}),
     (317, {'Department': 1, 'ManagementSalary': nan}),
     (318, {'Department': 10, 'ManagementSalary': 1.0}),
     (319, {'Department': 19, 'ManagementSalary': nan}),
     (320, {'Department': 7, 'ManagementSalary': nan}),
     (321, {'Department': 7, 'ManagementSalary': 0.0}),
     (322, {'Department': 17, 'ManagementSalary': 0.0}),
     (323, {'Department': 16, 'ManagementSalary': 0.0}),
     (324, {'Department': 14, 'ManagementSalary': 0.0}),
     (325, {'Department': 9, 'ManagementSalary': 0.0}),
     (326, {'Department': 9, 'ManagementSalary': 0.0}),
     (327, {'Department': 9, 'ManagementSalary': 1.0}),
     (328, {'Department': 8, 'ManagementSalary': 0.0}),
     (329, {'Department': 8, 'ManagementSalary': 1.0}),
     (330, {'Department': 13, 'ManagementSalary': 0.0}),
     (331, {'Department': 39, 'ManagementSalary': nan}),
     (332, {'Department': 14, 'ManagementSalary': 0.0}),
     (333, {'Department': 10, 'ManagementSalary': 1.0}),
     (334, {'Department': 17, 'ManagementSalary': 0.0}),
     (335, {'Department': 17, 'ManagementSalary': 0.0}),
     (336, {'Department': 13, 'ManagementSalary': 0.0}),
     (337, {'Department': 13, 'ManagementSalary': 1.0}),
     (338, {'Department': 13, 'ManagementSalary': 0.0}),
     (339, {'Department': 13, 'ManagementSalary': 0.0}),
     (340, {'Department': 2, 'ManagementSalary': 1.0}),
     (341, {'Department': 1, 'ManagementSalary': 0.0}),
     (342, {'Department': 0, 'ManagementSalary': nan}),
     (343, {'Department': 0, 'ManagementSalary': 0.0}),
     (344, {'Department': 0, 'ManagementSalary': 0.0}),
     (345, {'Department': 0, 'ManagementSalary': 0.0}),
     (346, {'Department': 0, 'ManagementSalary': 0.0}),
     (347, {'Department': 0, 'ManagementSalary': 0.0}),
     (348, {'Department': 0, 'ManagementSalary': 0.0}),
     (349, {'Department': 0, 'ManagementSalary': 0.0}),
     (350, {'Department': 0, 'ManagementSalary': 0.0}),
     (351, {'Department': 0, 'ManagementSalary': nan}),
     (352, {'Department': 0, 'ManagementSalary': nan}),
     (353, {'Department': 16, 'ManagementSalary': 0.0}),
     (354, {'Department': 16, 'ManagementSalary': 0.0}),
     (355, {'Department': 27, 'ManagementSalary': 1.0}),
     (356, {'Department': 8, 'ManagementSalary': nan}),
     (357, {'Department': 8, 'ManagementSalary': 0.0}),
     (358, {'Department': 14, 'ManagementSalary': 0.0}),
     (359, {'Department': 14, 'ManagementSalary': 0.0}),
     (360, {'Department': 14, 'ManagementSalary': 0.0}),
     (361, {'Department': 10, 'ManagementSalary': 1.0}),
     (362, {'Department': 14, 'ManagementSalary': nan}),
     (363, {'Department': 35, 'ManagementSalary': nan}),
     (364, {'Department': 37, 'ManagementSalary': 0.0}),
     (365, {'Department': 14, 'ManagementSalary': 0.0}),
     (366, {'Department': 36, 'ManagementSalary': nan}),
     (367, {'Department': 10, 'ManagementSalary': nan}),
     (368, {'Department': 7, 'ManagementSalary': 0.0}),
     (369, {'Department': 20, 'ManagementSalary': 0.0}),
     (370, {'Department': 10, 'ManagementSalary': 0.0}),
     (371, {'Department': 16, 'ManagementSalary': 0.0}),
     (372, {'Department': 36, 'ManagementSalary': nan}),
     (373, {'Department': 36, 'ManagementSalary': 0.0}),
     (374, {'Department': 14, 'ManagementSalary': 0.0}),
     (375, {'Department': 8, 'ManagementSalary': 0.0}),
     (376, {'Department': 7, 'ManagementSalary': 1.0}),
     (377, {'Department': 7, 'ManagementSalary': 1.0}),
     (378, {'Department': 7, 'ManagementSalary': 0.0}),
     (379, {'Department': 7, 'ManagementSalary': 1.0}),
     (380, {'Department': 7, 'ManagementSalary': nan}),
     (381, {'Department': 7, 'ManagementSalary': 0.0}),
     (382, {'Department': 7, 'ManagementSalary': nan}),
     (383, {'Department': 7, 'ManagementSalary': 0.0}),
     (384, {'Department': 7, 'ManagementSalary': nan}),
     (385, {'Department': 7, 'ManagementSalary': nan}),
     (386, {'Department': 7, 'ManagementSalary': nan}),
     (387, {'Department': 7, 'ManagementSalary': 0.0}),
     (388, {'Department': 7, 'ManagementSalary': 1.0}),
     (389, {'Department': 7, 'ManagementSalary': nan}),
     (390, {'Department': 7, 'ManagementSalary': 0.0}),
     (391, {'Department': 7, 'ManagementSalary': 0.0}),
     (392, {'Department': 7, 'ManagementSalary': 0.0}),
     (393, {'Department': 7, 'ManagementSalary': 0.0}),
     (394, {'Department': 7, 'ManagementSalary': 0.0}),
     (395, {'Department': 7, 'ManagementSalary': nan}),
     (396, {'Department': 7, 'ManagementSalary': 0.0}),
     (397, {'Department': 7, 'ManagementSalary': 1.0}),
     (398, {'Department': 7, 'ManagementSalary': 0.0}),
     (399, {'Department': 4, 'ManagementSalary': nan}),
     (400, {'Department': 9, 'ManagementSalary': 0.0}),
     (401, {'Department': 4, 'ManagementSalary': 0.0}),
     (402, {'Department': 0, 'ManagementSalary': nan}),
     (403, {'Department': 4, 'ManagementSalary': 0.0}),
     (404, {'Department': 16, 'ManagementSalary': 0.0}),
     (405, {'Department': 38, 'ManagementSalary': 1.0}),
     (406, {'Department': 14, 'ManagementSalary': nan}),
     (407, {'Department': 14, 'ManagementSalary': 0.0}),
     (408, {'Department': 21, 'ManagementSalary': nan}),
     (409, {'Department': 26, 'ManagementSalary': nan}),
     (410, {'Department': 27, 'ManagementSalary': 0.0}),
     (411, {'Department': 28, 'ManagementSalary': 1.0}),
     (412, {'Department': 21, 'ManagementSalary': nan}),
     (413, {'Department': 4, 'ManagementSalary': 0.0}),
     (414, {'Department': 1, 'ManagementSalary': 1.0}),
     (415, {'Department': 1, 'ManagementSalary': 0.0}),
     (416, {'Department': 9, 'ManagementSalary': 0.0}),
     (417, {'Department': 10, 'ManagementSalary': 1.0}),
     (418, {'Department': 15, 'ManagementSalary': 1.0}),
     (419, {'Department': 4, 'ManagementSalary': 1.0}),
     (420, {'Department': 26, 'ManagementSalary': nan}),
     (421, {'Department': 14, 'ManagementSalary': 0.0}),
     (422, {'Department': 35, 'ManagementSalary': 0.0}),
     (423, {'Department': 10, 'ManagementSalary': 1.0}),
     (424, {'Department': 34, 'ManagementSalary': 1.0}),
     (425, {'Department': 4, 'ManagementSalary': 0.0}),
     (426, {'Department': 4, 'ManagementSalary': 0.0}),
     (427, {'Department': 12, 'ManagementSalary': 0.0}),
     (428, {'Department': 17, 'ManagementSalary': 0.0}),
     (429, {'Department': 17, 'ManagementSalary': 0.0}),
     (430, {'Department': 14, 'ManagementSalary': 0.0}),
     (431, {'Department': 37, 'ManagementSalary': 0.0}),
     (432, {'Department': 37, 'ManagementSalary': 1.0}),
     (433, {'Department': 37, 'ManagementSalary': 0.0}),
     (434, {'Department': 34, 'ManagementSalary': nan}),
     (435, {'Department': 6, 'ManagementSalary': nan}),
     (436, {'Department': 13, 'ManagementSalary': 0.0}),
     (437, {'Department': 13, 'ManagementSalary': 0.0}),
     (438, {'Department': 13, 'ManagementSalary': 0.0}),
     (439, {'Department': 13, 'ManagementSalary': 0.0}),
     (440, {'Department': 4, 'ManagementSalary': 0.0}),
     (441, {'Department': 14, 'ManagementSalary': 0.0}),
     (442, {'Department': 10, 'ManagementSalary': 0.0}),
     (443, {'Department': 10, 'ManagementSalary': nan}),
     (444, {'Department': 10, 'ManagementSalary': 1.0}),
     (445, {'Department': 3, 'ManagementSalary': 0.0}),
     (446, {'Department': 17, 'ManagementSalary': 1.0}),
     (447, {'Department': 17, 'ManagementSalary': nan}),
     (448, {'Department': 17, 'ManagementSalary': 0.0}),
     (449, {'Department': 1, 'ManagementSalary': 0.0}),
     (450, {'Department': 4, 'ManagementSalary': 0.0}),
     (451, {'Department': 14, 'ManagementSalary': nan}),
     (452, {'Department': 14, 'ManagementSalary': 0.0}),
     (453, {'Department': 6, 'ManagementSalary': 1.0}),
     (454, {'Department': 27, 'ManagementSalary': 1.0}),
     (455, {'Department': 22, 'ManagementSalary': 0.0}),
     (456, {'Department': 21, 'ManagementSalary': nan}),
     (457, {'Department': 4, 'ManagementSalary': nan}),
     (458, {'Department': 4, 'ManagementSalary': nan}),
     (459, {'Department': 1, 'ManagementSalary': 0.0}),
     (460, {'Department': 34, 'ManagementSalary': 0.0}),
     (461, {'Department': 17, 'ManagementSalary': 0.0}),
     (462, {'Department': 30, 'ManagementSalary': 1.0}),
     (463, {'Department': 30, 'ManagementSalary': 0.0}),
     (464, {'Department': 4, 'ManagementSalary': 0.0}),
     (465, {'Department': 23, 'ManagementSalary': nan}),
     (466, {'Department': 14, 'ManagementSalary': nan}),
     (467, {'Department': 15, 'ManagementSalary': 0.0}),
     (468, {'Department': 1, 'ManagementSalary': 0.0}),
     (469, {'Department': 22, 'ManagementSalary': 0.0}),
     (470, {'Department': 12, 'ManagementSalary': 0.0}),
     (471, {'Department': 31, 'ManagementSalary': nan}),
     (472, {'Department': 6, 'ManagementSalary': 0.0}),
     (473, {'Department': 15, 'ManagementSalary': 0.0}),
     (474, {'Department': 15, 'ManagementSalary': 0.0}),
     (475, {'Department': 8, 'ManagementSalary': 0.0}),
     (476, {'Department': 15, 'ManagementSalary': 0.0}),
     (477, {'Department': 8, 'ManagementSalary': nan}),
     (478, {'Department': 8, 'ManagementSalary': 0.0}),
     (479, {'Department': 1, 'ManagementSalary': 0.0}),
     (480, {'Department': 15, 'ManagementSalary': 0.0}),
     (481, {'Department': 22, 'ManagementSalary': 1.0}),
     (482, {'Department': 2, 'ManagementSalary': nan}),
     (483, {'Department': 3, 'ManagementSalary': nan}),
     (484, {'Department': 4, 'ManagementSalary': nan}),
     (485, {'Department': 10, 'ManagementSalary': nan}),
     (486, {'Department': 4, 'ManagementSalary': 0.0}),
     (487, {'Department': 14, 'ManagementSalary': nan}),
     (488, {'Department': 14, 'ManagementSalary': 0.0}),
     (489, {'Department': 25, 'ManagementSalary': 1.0}),
     (490, {'Department': 6, 'ManagementSalary': 0.0}),
     (491, {'Department': 6, 'ManagementSalary': 0.0}),
     (492, {'Department': 40, 'ManagementSalary': nan}),
     (493, {'Department': 4, 'ManagementSalary': 1.0}),
     (494, {'Department': 36, 'ManagementSalary': nan}),
     (495, {'Department': 23, 'ManagementSalary': 1.0}),
     (496, {'Department': 14, 'ManagementSalary': nan}),
     (497, {'Department': 3, 'ManagementSalary': 0.0}),
     (498, {'Department': 14, 'ManagementSalary': 1.0}),
     (499, {'Department': 14, 'ManagementSalary': 0.0}),
     (500, {'Department': 14, 'ManagementSalary': nan}),
     (501, {'Department': 14, 'ManagementSalary': 0.0}),
     (502, {'Department': 14, 'ManagementSalary': 0.0}),
     (503, {'Department': 14, 'ManagementSalary': nan}),
     (504, {'Department': 14, 'ManagementSalary': 0.0}),
     (505, {'Department': 14, 'ManagementSalary': nan}),
     (506, {'Department': 14, 'ManagementSalary': 0.0}),
     (507, {'Department': 31, 'ManagementSalary': 0.0}),
     (508, {'Department': 15, 'ManagementSalary': 0.0}),
     (509, {'Department': 15, 'ManagementSalary': 1.0}),
     (510, {'Department': 14, 'ManagementSalary': 0.0}),
     (511, {'Department': 0, 'ManagementSalary': 0.0}),
     (512, {'Department': 23, 'ManagementSalary': 0.0}),
     (513, {'Department': 35, 'ManagementSalary': nan}),
     (514, {'Department': 8, 'ManagementSalary': 0.0}),
     (515, {'Department': 4, 'ManagementSalary': 0.0}),
     (516, {'Department': 1, 'ManagementSalary': nan}),
     (517, {'Department': 1, 'ManagementSalary': 0.0}),
     (518, {'Department': 35, 'ManagementSalary': nan}),
     (519, {'Department': 23, 'ManagementSalary': 0.0}),
     (520, {'Department': 21, 'ManagementSalary': nan}),
     (521, {'Department': 2, 'ManagementSalary': 0.0}),
     (522, {'Department': 4, 'ManagementSalary': nan}),
     (523, {'Department': 4, 'ManagementSalary': 0.0}),
     (524, {'Department': 9, 'ManagementSalary': nan}),
     (525, {'Department': 14, 'ManagementSalary': 0.0}),
     (526, {'Department': 4, 'ManagementSalary': 0.0}),
     (527, {'Department': 10, 'ManagementSalary': 1.0}),
     (528, {'Department': 25, 'ManagementSalary': 0.0}),
     (529, {'Department': 14, 'ManagementSalary': nan}),
     (530, {'Department': 14, 'ManagementSalary': nan}),
     (531, {'Department': 3, 'ManagementSalary': nan}),
     (532, {'Department': 21, 'ManagementSalary': 0.0}),
     (533, {'Department': 35, 'ManagementSalary': nan}),
     (534, {'Department': 4, 'ManagementSalary': 0.0}),
     (535, {'Department': 9, 'ManagementSalary': 0.0}),
     (536, {'Department': 15, 'ManagementSalary': 0.0}),
     (537, {'Department': 6, 'ManagementSalary': 0.0}),
     (538, {'Department': 9, 'ManagementSalary': nan}),
     (539, {'Department': 3, 'ManagementSalary': 0.0}),
     (540, {'Department': 15, 'ManagementSalary': 0.0}),
     (541, {'Department': 23, 'ManagementSalary': 0.0}),
     (542, {'Department': 4, 'ManagementSalary': 0.0}),
     (543, {'Department': 4, 'ManagementSalary': 1.0}),
     (544, {'Department': 4, 'ManagementSalary': 0.0}),
     (545, {'Department': 11, 'ManagementSalary': nan}),
     (546, {'Department': 35, 'ManagementSalary': 1.0}),
     (547, {'Department': 10, 'ManagementSalary': 0.0}),
     (548, {'Department': 6, 'ManagementSalary': 1.0}),
     (549, {'Department': 15, 'ManagementSalary': 0.0}),
     (550, {'Department': 15, 'ManagementSalary': 1.0}),
     (551, {'Department': 15, 'ManagementSalary': 0.0}),
     (552, {'Department': 22, 'ManagementSalary': 1.0}),
     (553, {'Department': 2, 'ManagementSalary': 0.0}),
     (554, {'Department': 2, 'ManagementSalary': 0.0}),
     (555, {'Department': 14, 'ManagementSalary': 0.0}),
     (556, {'Department': 4, 'ManagementSalary': 0.0}),
     (557, {'Department': 3, 'ManagementSalary': nan}),
     (558, {'Department': 14, 'ManagementSalary': 0.0}),
     (559, {'Department': 27, 'ManagementSalary': 0.0}),
     (560, {'Department': 31, 'ManagementSalary': 1.0}),
     (561, {'Department': 34, 'ManagementSalary': 0.0}),
     (562, {'Department': 4, 'ManagementSalary': 0.0}),
     (563, {'Department': 4, 'ManagementSalary': 0.0}),
     (564, {'Department': 19, 'ManagementSalary': 0.0}),
     (565, {'Department': 14, 'ManagementSalary': 0.0}),
     (566, {'Department': 14, 'ManagementSalary': 0.0}),
     (567, {'Department': 4, 'ManagementSalary': 0.0}),
     (568, {'Department': 4, 'ManagementSalary': 0.0}),
     (569, {'Department': 14, 'ManagementSalary': 0.0}),
     (570, {'Department': 14, 'ManagementSalary': 0.0}),
     (571, {'Department': 21, 'ManagementSalary': nan}),
     (572, {'Department': 4, 'ManagementSalary': 0.0}),
     (573, {'Department': 14, 'ManagementSalary': 1.0}),
     (574, {'Department': 4, 'ManagementSalary': 0.0}),
     (575, {'Department': 0, 'ManagementSalary': 0.0}),
     (576, {'Department': 4, 'ManagementSalary': 0.0}),
     (577, {'Department': 27, 'ManagementSalary': 0.0}),
     (578, {'Department': 27, 'ManagementSalary': nan}),
     (579, {'Department': 17, 'ManagementSalary': 0.0}),
     (580, {'Department': 16, 'ManagementSalary': 0.0}),
     (581, {'Department': 3, 'ManagementSalary': 0.0}),
     (582, {'Department': 15, 'ManagementSalary': nan}),
     (583, {'Department': 2, 'ManagementSalary': nan}),
     (584, {'Department': 4, 'ManagementSalary': 0.0}),
     (585, {'Department': 4, 'ManagementSalary': 0.0}),
     (586, {'Department': 21, 'ManagementSalary': 0.0}),
     (587, {'Department': 21, 'ManagementSalary': 0.0}),
     (588, {'Department': 11, 'ManagementSalary': 0.0}),
     (589, {'Department': 23, 'ManagementSalary': 0.0}),
     (590, {'Department': 11, 'ManagementSalary': 0.0}),
     (591, {'Department': 23, 'ManagementSalary': 0.0}),
     (592, {'Department': 17, 'ManagementSalary': 0.0}),
     (593, {'Department': 5, 'ManagementSalary': 0.0}),
     (594, {'Department': 36, 'ManagementSalary': 1.0}),
     (595, {'Department': 15, 'ManagementSalary': 0.0}),
     (596, {'Department': 23, 'ManagementSalary': 0.0}),
     (597, {'Department': 23, 'ManagementSalary': 0.0}),
     (598, {'Department': 2, 'ManagementSalary': 0.0}),
     (599, {'Department': 19, 'ManagementSalary': 0.0}),
     (600, {'Department': 4, 'ManagementSalary': 0.0}),
     (601, {'Department': 36, 'ManagementSalary': 0.0}),
     (602, {'Department': 14, 'ManagementSalary': nan}),
     (603, {'Department': 1, 'ManagementSalary': 0.0}),
     (604, {'Department': 22, 'ManagementSalary': nan}),
     (605, {'Department': 1, 'ManagementSalary': nan}),
     (606, {'Department': 21, 'ManagementSalary': 0.0}),
     (607, {'Department': 34, 'ManagementSalary': 0.0}),
     (608, {'Department': 14, 'ManagementSalary': 0.0}),
     (609, {'Department': 13, 'ManagementSalary': 0.0}),
     (610, {'Department': 6, 'ManagementSalary': 0.0}),
     (611, {'Department': 4, 'ManagementSalary': 0.0}),
     (612, {'Department': 37, 'ManagementSalary': nan}),
     (613, {'Department': 6, 'ManagementSalary': nan}),
     (614, {'Department': 24, 'ManagementSalary': 0.0}),
     (615, {'Department': 35, 'ManagementSalary': nan}),
     (616, {'Department': 6, 'ManagementSalary': 0.0}),
     (617, {'Department': 17, 'ManagementSalary': 0.0}),
     (618, {'Department': 16, 'ManagementSalary': 0.0}),
     (619, {'Department': 6, 'ManagementSalary': 0.0}),
     (620, {'Department': 4, 'ManagementSalary': 0.0}),
     (621, {'Department': 0, 'ManagementSalary': 0.0}),
     (622, {'Department': 21, 'ManagementSalary': 0.0}),
     (623, {'Department': 4, 'ManagementSalary': 0.0}),
     (624, {'Department': 26, 'ManagementSalary': 0.0}),
     (625, {'Department': 21, 'ManagementSalary': nan}),
     (626, {'Department': 4, 'ManagementSalary': 0.0}),
     (627, {'Department': 15, 'ManagementSalary': 0.0}),
     (628, {'Department': 7, 'ManagementSalary': 0.0}),
     (629, {'Department': 1, 'ManagementSalary': 0.0}),
     (630, {'Department': 20, 'ManagementSalary': 0.0}),
     (631, {'Department': 19, 'ManagementSalary': 0.0}),
     (632, {'Department': 7, 'ManagementSalary': 0.0}),
     (633, {'Department': 21, 'ManagementSalary': 0.0}),
     (634, {'Department': 21, 'ManagementSalary': 0.0}),
     (635, {'Department': 21, 'ManagementSalary': 0.0}),
     (636, {'Department': 21, 'ManagementSalary': nan}),
     (637, {'Department': 19, 'ManagementSalary': 0.0}),
     (638, {'Department': 38, 'ManagementSalary': 0.0}),
     (639, {'Department': 19, 'ManagementSalary': 0.0}),
     (640, {'Department': 16, 'ManagementSalary': nan}),
     (641, {'Department': 23, 'ManagementSalary': nan}),
     (642, {'Department': 6, 'ManagementSalary': 0.0}),
     (643, {'Department': 37, 'ManagementSalary': 0.0}),
     (644, {'Department': 25, 'ManagementSalary': 0.0}),
     (645, {'Department': 1, 'ManagementSalary': 0.0}),
     (646, {'Department': 22, 'ManagementSalary': nan}),
     (647, {'Department': 6, 'ManagementSalary': nan}),
     (648, {'Department': 21, 'ManagementSalary': 0.0}),
     (649, {'Department': 14, 'ManagementSalary': 0.0}),
     (650, {'Department': 1, 'ManagementSalary': 0.0}),
     (651, {'Department': 26, 'ManagementSalary': nan}),
     (652, {'Department': 8, 'ManagementSalary': 0.0}),
     (653, {'Department': 7, 'ManagementSalary': 0.0}),
     (654, {'Department': 37, 'ManagementSalary': 0.0}),
     (655, {'Department': 4, 'ManagementSalary': nan}),
     (656, {'Department': 0, 'ManagementSalary': nan}),
     (657, {'Department': 17, 'ManagementSalary': 0.0}),
     (658, {'Department': 14, 'ManagementSalary': 0.0}),
     (659, {'Department': 6, 'ManagementSalary': 0.0}),
     (660, {'Department': 17, 'ManagementSalary': 0.0}),
     (661, {'Department': 14, 'ManagementSalary': 0.0}),
     (662, {'Department': 16, 'ManagementSalary': 0.0}),
     (663, {'Department': 15, 'ManagementSalary': 0.0}),
     (664, {'Department': 4, 'ManagementSalary': nan}),
     (665, {'Department': 32, 'ManagementSalary': 0.0}),
     (666, {'Department': 14, 'ManagementSalary': nan}),
     (667, {'Department': 15, 'ManagementSalary': 0.0}),
     (668, {'Department': 0, 'ManagementSalary': 0.0}),
     (669, {'Department': 23, 'ManagementSalary': nan}),
     (670, {'Department': 21, 'ManagementSalary': nan}),
     (671, {'Department': 29, 'ManagementSalary': nan}),
     (672, {'Department': 14, 'ManagementSalary': 0.0}),
     (673, {'Department': 23, 'ManagementSalary': 0.0}),
     (674, {'Department': 14, 'ManagementSalary': 0.0}),
     (675, {'Department': 1, 'ManagementSalary': 0.0}),
     (676, {'Department': 17, 'ManagementSalary': nan}),
     (677, {'Department': 26, 'ManagementSalary': 0.0}),
     (678, {'Department': 15, 'ManagementSalary': nan}),
     (679, {'Department': 29, 'ManagementSalary': 0.0}),
     (680, {'Department': 0, 'ManagementSalary': 0.0}),
     (681, {'Department': 0, 'ManagementSalary': 0.0}),
     (682, {'Department': 0, 'ManagementSalary': nan}),
     (683, {'Department': 22, 'ManagementSalary': nan}),
     (684, {'Department': 34, 'ManagementSalary': 0.0}),
     (685, {'Department': 21, 'ManagementSalary': nan}),
     (686, {'Department': 6, 'ManagementSalary': 0.0}),
     (687, {'Department': 16, 'ManagementSalary': 0.0}),
     (688, {'Department': 4, 'ManagementSalary': 0.0}),
     (689, {'Department': 4, 'ManagementSalary': 0.0}),
     (690, {'Department': 15, 'ManagementSalary': 0.0}),
     (691, {'Department': 21, 'ManagementSalary': nan}),
     (692, {'Department': 0, 'ManagementSalary': 0.0}),
     (693, {'Department': 36, 'ManagementSalary': 0.0}),
     (694, {'Department': 4, 'ManagementSalary': 0.0}),
     (695, {'Department': 23, 'ManagementSalary': 0.0}),
     (696, {'Department': 1, 'ManagementSalary': 0.0}),
     (697, {'Department': 1, 'ManagementSalary': 0.0}),
     (698, {'Department': 22, 'ManagementSalary': 0.0}),
     (699, {'Department': 14, 'ManagementSalary': 0.0}),
     (700, {'Department': 14, 'ManagementSalary': 0.0}),
     (701, {'Department': 30, 'ManagementSalary': 0.0}),
     (702, {'Department': 4, 'ManagementSalary': 0.0}),
     (703, {'Department': 9, 'ManagementSalary': 0.0}),
     (704, {'Department': 10, 'ManagementSalary': 0.0}),
     (705, {'Department': 4, 'ManagementSalary': 0.0}),
     (706, {'Department': 4, 'ManagementSalary': 0.0}),
     (707, {'Department': 14, 'ManagementSalary': 0.0}),
     (708, {'Department': 16, 'ManagementSalary': nan}),
     (709, {'Department': 16, 'ManagementSalary': 0.0}),
     (710, {'Department': 15, 'ManagementSalary': nan}),
     (711, {'Department': 21, 'ManagementSalary': 0.0}),
     (712, {'Department': 0, 'ManagementSalary': 0.0}),
     (713, {'Department': 15, 'ManagementSalary': nan}),
     (714, {'Department': 4, 'ManagementSalary': 0.0}),
     (715, {'Department': 15, 'ManagementSalary': 0.0}),
     (716, {'Department': 29, 'ManagementSalary': 0.0}),
     (717, {'Department': 24, 'ManagementSalary': nan}),
     (718, {'Department': 21, 'ManagementSalary': 0.0}),
     (719, {'Department': 7, 'ManagementSalary': 0.0}),
     (720, {'Department': 14, 'ManagementSalary': nan}),
     (721, {'Department': 11, 'ManagementSalary': 0.0}),
     (722, {'Department': 11, 'ManagementSalary': 0.0}),
     (723, {'Department': 9, 'ManagementSalary': 0.0}),
     (724, {'Department': 13, 'ManagementSalary': 0.0}),
     (725, {'Department': 10, 'ManagementSalary': nan}),
     (726, {'Department': 31, 'ManagementSalary': 0.0}),
     (727, {'Department': 4, 'ManagementSalary': 0.0}),
     (728, {'Department': 22, 'ManagementSalary': 0.0}),
     (729, {'Department': 14, 'ManagementSalary': nan}),
     (730, {'Department': 23, 'ManagementSalary': 0.0}),
     (731, {'Department': 1, 'ManagementSalary': 0.0}),
     (732, {'Department': 4, 'ManagementSalary': 0.0}),
     (733, {'Department': 9, 'ManagementSalary': 0.0}),
     (734, {'Department': 1, 'ManagementSalary': 0.0}),
     (735, {'Department': 17, 'ManagementSalary': 0.0}),
     (736, {'Department': 27, 'ManagementSalary': nan}),
     (737, {'Department': 28, 'ManagementSalary': 0.0}),
     (738, {'Department': 22, 'ManagementSalary': nan}),
     (739, {'Department': 14, 'ManagementSalary': 0.0}),
     (740, {'Department': 20, 'ManagementSalary': 0.0}),
     (741, {'Department': 7, 'ManagementSalary': nan}),
     (742, {'Department': 23, 'ManagementSalary': nan}),
     (743, {'Department': 1, 'ManagementSalary': nan}),
     (744, {'Department': 4, 'ManagementSalary': 0.0}),
     (745, {'Department': 6, 'ManagementSalary': 0.0}),
     (746, {'Department': 15, 'ManagementSalary': 0.0}),
     (747, {'Department': 15, 'ManagementSalary': 1.0}),
     (748, {'Department': 23, 'ManagementSalary': nan}),
     (749, {'Department': 4, 'ManagementSalary': 0.0}),
     (750, {'Department': 20, 'ManagementSalary': 0.0}),
     (751, {'Department': 5, 'ManagementSalary': 0.0}),
     (752, {'Department': 36, 'ManagementSalary': 0.0}),
     (753, {'Department': 10, 'ManagementSalary': 0.0}),
     (754, {'Department': 14, 'ManagementSalary': 0.0}),
     (755, {'Department': 21, 'ManagementSalary': 0.0}),
     (756, {'Department': 39, 'ManagementSalary': 0.0}),
     (757, {'Department': 10, 'ManagementSalary': 0.0}),
     (758, {'Department': 41, 'ManagementSalary': nan}),
     (759, {'Department': 31, 'ManagementSalary': 0.0}),
     (760, {'Department': 17, 'ManagementSalary': nan}),
     (761, {'Department': 7, 'ManagementSalary': 0.0}),
     (762, {'Department': 21, 'ManagementSalary': 0.0}),
     (763, {'Department': 34, 'ManagementSalary': 0.0}),
     (764, {'Department': 1, 'ManagementSalary': nan}),
     (765, {'Department': 14, 'ManagementSalary': nan}),
     (766, {'Department': 2, 'ManagementSalary': nan}),
     (767, {'Department': 18, 'ManagementSalary': 0.0}),
     (768, {'Department': 16, 'ManagementSalary': nan}),
     (769, {'Department': 27, 'ManagementSalary': 0.0}),
     (770, {'Department': 16, 'ManagementSalary': 0.0}),
     (771, {'Department': 38, 'ManagementSalary': 0.0}),
     (772, {'Department': 7, 'ManagementSalary': 0.0}),
     (773, {'Department': 38, 'ManagementSalary': nan}),
     (774, {'Department': 21, 'ManagementSalary': 0.0}),
     (775, {'Department': 1, 'ManagementSalary': 0.0}),
     (776, {'Department': 5, 'ManagementSalary': nan}),
     (777, {'Department': 9, 'ManagementSalary': 0.0}),
     (778, {'Department': 15, 'ManagementSalary': 0.0}),
     (779, {'Department': 15, 'ManagementSalary': 0.0}),
     (780, {'Department': 15, 'ManagementSalary': 0.0}),
     (781, {'Department': 0, 'ManagementSalary': 0.0}),
     (782, {'Department': 6, 'ManagementSalary': 1.0}),
     (783, {'Department': 23, 'ManagementSalary': nan}),
     (784, {'Department': 28, 'ManagementSalary': 0.0}),
     (785, {'Department': 11, 'ManagementSalary': 0.0}),
     (786, {'Department': 23, 'ManagementSalary': nan}),
     (787, {'Department': 34, 'ManagementSalary': nan}),
     (788, {'Department': 24, 'ManagementSalary': nan}),
     (789, {'Department': 4, 'ManagementSalary': nan}),
     (790, {'Department': 4, 'ManagementSalary': 0.0}),
     (791, {'Department': 4, 'ManagementSalary': 0.0}),
     (792, {'Department': 24, 'ManagementSalary': 0.0}),
     (793, {'Department': 23, 'ManagementSalary': 0.0}),
     (794, {'Department': 17, 'ManagementSalary': 0.0}),
     (795, {'Department': 10, 'ManagementSalary': 0.0}),
     (796, {'Department': 17, 'ManagementSalary': 0.0}),
     (797, {'Department': 1, 'ManagementSalary': 0.0}),
     (798, {'Department': 1, 'ManagementSalary': nan}),
     (799, {'Department': 15, 'ManagementSalary': nan}),
     (800, {'Department': 15, 'ManagementSalary': nan}),
     (801, {'Department': 4, 'ManagementSalary': 0.0}),
     (802, {'Department': 4, 'ManagementSalary': 0.0}),
     (803, {'Department': 21, 'ManagementSalary': 0.0}),
     (804, {'Department': 14, 'ManagementSalary': 0.0}),
     (805, {'Department': 14, 'ManagementSalary': 0.0}),
     (806, {'Department': 20, 'ManagementSalary': 0.0}),
     (807, {'Department': 28, 'ManagementSalary': 0.0}),
     (808, {'Department': 20, 'ManagementSalary': nan}),
     (809, {'Department': 22, 'ManagementSalary': 1.0}),
     (810, {'Department': 26, 'ManagementSalary': 0.0}),
     (811, {'Department': 3, 'ManagementSalary': 0.0}),
     (812, {'Department': 32, 'ManagementSalary': 0.0}),
     (813, {'Department': 4, 'ManagementSalary': 0.0}),
     (814, {'Department': 0, 'ManagementSalary': 0.0}),
     (815, {'Department': 21, 'ManagementSalary': 0.0}),
     (816, {'Department': 13, 'ManagementSalary': 0.0}),
     (817, {'Department': 4, 'ManagementSalary': nan}),
     (818, {'Department': 15, 'ManagementSalary': nan}),
     (819, {'Department': 17, 'ManagementSalary': nan}),
     (820, {'Department': 5, 'ManagementSalary': nan}),
     (821, {'Department': 24, 'ManagementSalary': 1.0}),
     (822, {'Department': 4, 'ManagementSalary': nan}),
     (823, {'Department': 14, 'ManagementSalary': 0.0}),
     (824, {'Department': 0, 'ManagementSalary': 0.0}),
     (825, {'Department': 9, 'ManagementSalary': nan}),
     (826, {'Department': 21, 'ManagementSalary': nan}),
     (827, {'Department': 14, 'ManagementSalary': 0.0}),
     (828, {'Department': 38, 'ManagementSalary': 1.0}),
     (829, {'Department': 4, 'ManagementSalary': nan}),
     (830, {'Department': 14, 'ManagementSalary': 0.0}),
     (831, {'Department': 31, 'ManagementSalary': 0.0}),
     (832, {'Department': 21, 'ManagementSalary': 0.0}),
     (833, {'Department': 14, 'ManagementSalary': 0.0}),
     (834, {'Department': 6, 'ManagementSalary': 0.0}),
     (835, {'Department': 4, 'ManagementSalary': 0.0}),
     (836, {'Department': 4, 'ManagementSalary': 0.0}),
     (837, {'Department': 6, 'ManagementSalary': nan}),
     (838, {'Department': 17, 'ManagementSalary': 0.0}),
     (839, {'Department': 0, 'ManagementSalary': 0.0}),
     (840, {'Department': 4, 'ManagementSalary': 1.0}),
     (841, {'Department': 7, 'ManagementSalary': 0.0}),
     (842, {'Department': 16, 'ManagementSalary': 0.0}),
     (843, {'Department': 4, 'ManagementSalary': 0.0}),
     (844, {'Department': 4, 'ManagementSalary': 0.0}),
     (845, {'Department': 21, 'ManagementSalary': 0.0}),
     (846, {'Department': 1, 'ManagementSalary': 0.0}),
     (847, {'Department': 10, 'ManagementSalary': 0.0}),
     (848, {'Department': 3, 'ManagementSalary': 0.0}),
     (849, {'Department': 21, 'ManagementSalary': 0.0}),
     (850, {'Department': 4, 'ManagementSalary': 0.0}),
     (851, {'Department': 0, 'ManagementSalary': 0.0}),
     (852, {'Department': 1, 'ManagementSalary': 0.0}),
     (853, {'Department': 7, 'ManagementSalary': 0.0}),
     (854, {'Department': 17, 'ManagementSalary': 0.0}),
     (855, {'Department': 15, 'ManagementSalary': 0.0}),
     (856, {'Department': 14, 'ManagementSalary': 0.0}),
     (857, {'Department': 0, 'ManagementSalary': 0.0}),
     (858, {'Department': 9, 'ManagementSalary': nan}),
     (859, {'Department': 32, 'ManagementSalary': 1.0}),
     (860, {'Department': 13, 'ManagementSalary': 0.0}),
     (861, {'Department': 5, 'ManagementSalary': 0.0}),
     (862, {'Department': 2, 'ManagementSalary': 0.0}),
     (863, {'Department': 21, 'ManagementSalary': nan}),
     (864, {'Department': 28, 'ManagementSalary': nan}),
     (865, {'Department': 21, 'ManagementSalary': nan}),
     (866, {'Department': 22, 'ManagementSalary': nan}),
     (867, {'Department': 22, 'ManagementSalary': nan}),
     (868, {'Department': 7, 'ManagementSalary': 0.0}),
     (869, {'Department': 7, 'ManagementSalary': 0.0}),
     (870, {'Department': 33, 'ManagementSalary': 0.0}),
     (871, {'Department': 0, 'ManagementSalary': nan}),
     (872, {'Department': 1, 'ManagementSalary': 0.0}),
     (873, {'Department': 15, 'ManagementSalary': 0.0}),
     (874, {'Department': 4, 'ManagementSalary': 0.0}),
     (875, {'Department': 31, 'ManagementSalary': 0.0}),
     (876, {'Department': 30, 'ManagementSalary': 0.0}),
     (877, {'Department': 15, 'ManagementSalary': nan}),
     (878, {'Department': 11, 'ManagementSalary': 0.0}),
     (879, {'Department': 19, 'ManagementSalary': nan}),
     (880, {'Department': 21, 'ManagementSalary': 1.0}),
     (881, {'Department': 9, 'ManagementSalary': 0.0}),
     (882, {'Department': 21, 'ManagementSalary': 1.0}),
     (883, {'Department': 13, 'ManagementSalary': 0.0}),
     (884, {'Department': 21, 'ManagementSalary': 0.0}),
     (885, {'Department': 9, 'ManagementSalary': 0.0}),
     (886, {'Department': 32, 'ManagementSalary': nan}),
     (887, {'Department': 9, 'ManagementSalary': 0.0}),
     (888, {'Department': 32, 'ManagementSalary': 0.0}),
     (889, {'Department': 38, 'ManagementSalary': 0.0}),
     (890, {'Department': 9, 'ManagementSalary': nan}),
     (891, {'Department': 38, 'ManagementSalary': 0.0}),
     (892, {'Department': 38, 'ManagementSalary': nan}),
     (893, {'Department': 14, 'ManagementSalary': 0.0}),
     (894, {'Department': 9, 'ManagementSalary': 0.0}),
     (895, {'Department': 10, 'ManagementSalary': 1.0}),
     (896, {'Department': 38, 'ManagementSalary': 0.0}),
     (897, {'Department': 10, 'ManagementSalary': 0.0}),
     (898, {'Department': 22, 'ManagementSalary': 0.0}),
     (899, {'Department': 21, 'ManagementSalary': nan}),
     (900, {'Department': 13, 'ManagementSalary': 0.0}),
     (901, {'Department': 21, 'ManagementSalary': 0.0}),
     (902, {'Department': 4, 'ManagementSalary': 0.0}),
     (903, {'Department': 0, 'ManagementSalary': 0.0}),
     (904, {'Department': 1, 'ManagementSalary': 0.0}),
     (905, {'Department': 1, 'ManagementSalary': 0.0}),
     (906, {'Department': 23, 'ManagementSalary': 0.0}),
     (907, {'Department': 0, 'ManagementSalary': 0.0}),
     (908, {'Department': 5, 'ManagementSalary': 0.0}),
     (909, {'Department': 4, 'ManagementSalary': 0.0}),
     (910, {'Department': 4, 'ManagementSalary': 0.0}),
     (911, {'Department': 15, 'ManagementSalary': nan}),
     (912, {'Department': 14, 'ManagementSalary': nan}),
     (913, {'Department': 14, 'ManagementSalary': nan}),
     (914, {'Department': 13, 'ManagementSalary': nan}),
     (915, {'Department': 11, 'ManagementSalary': nan}),
     (916, {'Department': 1, 'ManagementSalary': 0.0}),
     (917, {'Department': 5, 'ManagementSalary': 0.0}),
     (918, {'Department': 5, 'ManagementSalary': nan}),
     (919, {'Department': 10, 'ManagementSalary': 0.0}),
     (920, {'Department': 23, 'ManagementSalary': 0.0}),
     (921, {'Department': 21, 'ManagementSalary': 0.0}),
     (922, {'Department': 14, 'ManagementSalary': 0.0}),
     (923, {'Department': 9, 'ManagementSalary': nan}),
     (924, {'Department': 20, 'ManagementSalary': 0.0}),
     (925, {'Department': 10, 'ManagementSalary': 1.0}),
     (926, {'Department': 19, 'ManagementSalary': nan}),
     (927, {'Department': 19, 'ManagementSalary': 0.0}),
     (928, {'Department': 21, 'ManagementSalary': 0.0}),
     (929, {'Department': 17, 'ManagementSalary': 0.0}),
     (930, {'Department': 19, 'ManagementSalary': 0.0}),
     (931, {'Department': 19, 'ManagementSalary': nan}),
     (932, {'Department': 36, 'ManagementSalary': 0.0}),
     (933, {'Department': 17, 'ManagementSalary': 0.0}),
     (934, {'Department': 35, 'ManagementSalary': nan}),
     (935, {'Department': 16, 'ManagementSalary': 0.0}),
     (936, {'Department': 4, 'ManagementSalary': 0.0}),
     (937, {'Department': 16, 'ManagementSalary': 0.0}),
     (938, {'Department': 4, 'ManagementSalary': 0.0}),
     (939, {'Department': 6, 'ManagementSalary': nan}),
     (940, {'Department': 4, 'ManagementSalary': 0.0}),
     (941, {'Department': 41, 'ManagementSalary': 0.0}),
     (942, {'Department': 6, 'ManagementSalary': 0.0}),
     (943, {'Department': 7, 'ManagementSalary': 0.0}),
     (944, {'Department': 23, 'ManagementSalary': nan}),
     (945, {'Department': 9, 'ManagementSalary': nan}),
     (946, {'Department': 23, 'ManagementSalary': 0.0}),
     (947, {'Department': 7, 'ManagementSalary': nan}),
     (948, {'Department': 6, 'ManagementSalary': 0.0}),
     (949, {'Department': 22, 'ManagementSalary': 0.0}),
     (950, {'Department': 36, 'ManagementSalary': nan}),
     (951, {'Department': 14, 'ManagementSalary': nan}),
     (952, {'Department': 15, 'ManagementSalary': 0.0}),
     (953, {'Department': 11, 'ManagementSalary': nan}),
     (954, {'Department': 35, 'ManagementSalary': 0.0}),
     (955, {'Department': 5, 'ManagementSalary': 0.0}),
     (956, {'Department': 14, 'ManagementSalary': 0.0}),
     (957, {'Department': 14, 'ManagementSalary': 0.0}),
     (958, {'Department': 15, 'ManagementSalary': 0.0}),
     (959, {'Department': 4, 'ManagementSalary': nan}),
     (960, {'Department': 6, 'ManagementSalary': 0.0}),
     (961, {'Department': 4, 'ManagementSalary': 0.0}),
     (962, {'Department': 9, 'ManagementSalary': nan}),
     (963, {'Department': 19, 'ManagementSalary': nan}),
     (964, {'Department': 11, 'ManagementSalary': 0.0}),
     (965, {'Department': 4, 'ManagementSalary': 0.0}),
     (966, {'Department': 29, 'ManagementSalary': 0.0}),
     (967, {'Department': 14, 'ManagementSalary': 0.0}),
     (968, {'Department': 15, 'ManagementSalary': nan}),
     (969, {'Department': 15, 'ManagementSalary': nan}),
     (970, {'Department': 5, 'ManagementSalary': 0.0}),
     (971, {'Department': 32, 'ManagementSalary': 1.0}),
     (972, {'Department': 15, 'ManagementSalary': 0.0}),
     (973, {'Department': 14, 'ManagementSalary': 0.0}),
     (974, {'Department': 5, 'ManagementSalary': nan}),
     (975, {'Department': 9, 'ManagementSalary': 0.0}),
     (976, {'Department': 10, 'ManagementSalary': 0.0}),
     (977, {'Department': 19, 'ManagementSalary': 0.0}),
     (978, {'Department': 13, 'ManagementSalary': 0.0}),
     (979, {'Department': 23, 'ManagementSalary': 0.0}),
     (980, {'Department': 12, 'ManagementSalary': 0.0}),
     (981, {'Department': 10, 'ManagementSalary': 0.0}),
     (982, {'Department': 21, 'ManagementSalary': 0.0}),
     (983, {'Department': 10, 'ManagementSalary': 0.0}),
     (984, {'Department': 35, 'ManagementSalary': nan}),
     (985, {'Department': 7, 'ManagementSalary': 0.0}),
     (986, {'Department': 22, 'ManagementSalary': 0.0}),
     (987, {'Department': 22, 'ManagementSalary': nan}),
     (988, {'Department': 22, 'ManagementSalary': 0.0}),
     (989, {'Department': 8, 'ManagementSalary': nan}),
     (990, {'Department': 21, 'ManagementSalary': 0.0}),
     (991, {'Department': 32, 'ManagementSalary': nan}),
     (992, {'Department': 4, 'ManagementSalary': nan}),
     (993, {'Department': 21, 'ManagementSalary': 0.0}),
     (994, {'Department': 21, 'ManagementSalary': nan}),
     (995, {'Department': 6, 'ManagementSalary': 0.0}),
     (996, {'Department': 14, 'ManagementSalary': nan}),
     (997, {'Department': 11, 'ManagementSalary': 0.0}),
     (998, {'Department': 14, 'ManagementSalary': 0.0}),
     (999, {'Department': 15, 'ManagementSalary': 0.0}),
     ...]




```python
dfG = DataFrame(index=G.nodes())

dfG['Department'] = pd.Series(nx.get_node_attributes(G, 'Department'))
dfG['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
dfG
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Department</th>
      <th>ManagementSalary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>25</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>14</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>34</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>14</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>11</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>975</th>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>976</th>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>977</th>
      <td>19</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>978</th>
      <td>13</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>979</th>
      <td>23</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>980</th>
      <td>12</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>981</th>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>982</th>
      <td>21</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>983</th>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>984</th>
      <td>35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>985</th>
      <td>7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>986</th>
      <td>22</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>987</th>
      <td>22</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>988</th>
      <td>22</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>989</th>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>990</th>
      <td>21</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>991</th>
      <td>32</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>992</th>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>993</th>
      <td>21</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>994</th>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>995</th>
      <td>6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>14</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>997</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>14</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>15</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>22</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1005 rows × 2 columns</p>
</div>




```python
dfG.ManagementSalary.value_counts(dropna=False)
```




     0.0    634
    NaN     252
     1.0    119
    Name: ManagementSalary, dtype: int64



Ideas for features to generate to then train on:
- Degree (normal): presumption that managers send emails to more people than non-managers
- Degree out of department: assumption is managers communicate laterally
- Degree within department (normalised): assumption managers communicate to every person in department


```python
dfG.Department.value_counts()
```




    4     109
    14     92
    1      65
    21     61
    15     55
    7      51
    0      49
    10     39
    17     35
    9      32
    19     29
    11     29
    6      28
    23     27
    13     26
    16     25
    22     25
    36     22
    8      19
    5      18
    37     15
    20     14
    34     13
    38     13
    35     13
    3      12
    27     10
    2      10
    26      9
    32      9
    31      8
    28      8
    24      6
    25      6
    29      5
    40      4
    30      4
    12      3
    39      3
    41      2
    33      1
    18      1
    Name: Department, dtype: int64




```python
dfG.iloc[[33,18]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Department</th>
      <th>ManagementSalary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>11</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add degrees to each node
dfG['degree'] = pd.Series(G.degree())
dfG['degCent'] = pd.Series(nx.degree_centrality(G))
print('Average degree of management is {0}\nAverage degree of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].degree.mean(), dfG[dfG.ManagementSalary==0.0].degree.mean()))
print('Average degree centrality of management is {0}\nAverage degree centrality of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].degCent.mean(), dfG[dfG.ManagementSalary==0.0].degCent.mean()))
```

    Average degree of management is 81.94117647058823
    Average degree of non-management is 24.613564668769715
    Average degree centrality of management is 0.08161471760018753
    Average degree centrality of non-management is 0.024515502658137307
    


```python
# Add local clustering coefficient for each node
dfG['LCC'] = pd.Series(nx.clustering(G))
print('Average LCC of management is {0}\nAverage LCC of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].LCC.mean(), dfG[dfG.ManagementSalary==0.0].LCC.mean()))
```

    Average LCC of management is 0.3823610869906431
    Average LCC of non-management is 0.3975504930090915
    


```python
nx.closeness_centrality(G, normalized=True)
```




    {0: 0.42199107500130484,
     1: 0.4223599483186137,
     2: 0.46148976205968867,
     3: 0.44166341944834925,
     4: 0.46215187075704833,
     5: 0.5014839448640311,
     6: 0.4758048063776406,
     7: 0.42015633119695134,
     8: 0.41315073183111933,
     9: 0.3561959313501615,
     10: 0.4424723268099762,
     11: 0.4312180105992807,
     12: 0.43315085690407357,
     13: 0.503050266399265,
     14: 0.44206750308919857,
     15: 0.4298752498901192,
     16: 0.45799031362700854,
     17: 0.46979074465385906,
     18: 0.44206750308919857,
     19: 0.4414616545239781,
     20: 0.45199231139054635,
     21: 0.4878140140095851,
     22: 0.3792619944085511,
     23: 0.45135897326155444,
     24: 0.39817040039266094,
     25: 0.4138584846907872,
     26: 0.3815079201551473,
     27: 0.43025804174220306,
     28: 0.46729185771421083,
     29: 0.4414616545239781,
     30: 0.4432842026389854,
     31: 0.418156452511029,
     32: 0.38747376172934567,
     33: 0.39346887693525573,
     34: 0.38256514717062073,
     35: 0.4418653688856827,
     36: 0.4325691861025014,
     37: 0.38950405552317136,
     38: 0.4072311680375002,
     39: 0.4039964723047609,
     40: 0.4152812899669051,
     41: 0.46148976205968867,
     42: 0.4606098959737789,
     43: 0.3308317568479932,
     44: 0.4648194140226013,
     45: 0.41979129528800524,
     46: 0.4480109233903514,
     47: 0.4378611516778378,
     48: 0.4120936297454107,
     49: 0.36085121798095143,
     50: 0.3736889256585414,
     51: 0.44759590632375545,
     52: 0.3870082345826945,
     53: 0.42873095020097074,
     54: 0.41869998342850434,
     55: 0.4236560989710601,
     56: 0.4124453955411814,
     57: 0.42968410927211564,
     58: 0.47768638742115077,
     59: 0.41456866656069846,
     60: 0.405352165164844,
     61: 0.4372667700239765,
     62: 0.5226390274488848,
     63: 0.47934502071080753,
     64: 0.5067433464881951,
     65: 0.46104940923329585,
     66: 0.4210717044675329,
     67: 0.39621138243254944,
     68: 0.42968410927211564,
     69: 0.4366739998883814,
     70: 0.39060612843693937,
     71: 0.3760153936782055,
     72: 0.3467382711707887,
     73: 0.3994872103154147,
     74: 0.45199231139054635,
     75: 0.3749940092173023,
     76: 0.37572300223677607,
     77: 0.4018127075896,
     78: 0.41492467228552515,
     79: 0.43295679290008426,
     80: 0.44926060518502464,
     81: 0.4755706504689902,
     82: 0.5341954459662731,
     83: 0.49304059273111633,
     84: 0.48221535017614175,
     85: 0.41226943760793006,
     86: 0.5220743175326786,
     87: 0.49354420927118897,
     88: 0.4120936297454107,
     89: 0.4451218617010539,
     90: 0.4210717044675329,
     91: 0.4108671606092636,
     92: 0.4300665606377339,
     93: 0.4515698886696206,
     94: 0.4212552579568387,
     95: 0.4244003345423751,
     96: 0.4675179302143145,
     97: 0.38302004033015774,
     98: 0.3976788319971144,
     99: 0.4043345446665222,
     100: 0.41851864952489737,
     101: 0.3960490007184377,
     102: 0.3840856763724118,
     103: 0.43179605082796607,
     104: 0.40791876815237993,
     105: 0.4965876473550812,
     106: 0.48954385093869707,
     107: 0.5237721201913215,
     108: 0.42458680217618106,
     109: 0.36085121798095143,
     110: 0.3648016465658694,
     111: 0.3456221608558612,
     112: 0.3786675398718605,
     113: 0.4588601907659013,
     114: 0.4897919725053158,
     115: 0.48221535017614175,
     116: 0.44842671079024965,
     117: 0.4188814745353221,
     118: 0.418156452511029,
     119: 0.3596425611287637,
     120: 0.39171445551397976,
     121: 0.5312586925524948,
     122: 0.3739781585731378,
     123: 0.4210717044675329,
     124: 0.40791876815237993,
     125: 0.4051822061857392,
     126: 0.3955626531940189,
     127: 0.42892124356546296,
     128: 0.49633259463430307,
     129: 0.5038371020609949,
     130: 0.422175431084748,
     131: 0.440455588766175,
     132: 0.444303246782983,
     133: 0.4755706504689902,
     134: 0.4039964723047609,
     135: 0.41942689312195663,
     136: 0.40214713348022807,
     137: 0.4549715450814445,
     138: 0.4124453955411814,
     139: 0.4172536967845371,
     140: 0.41545982878460364,
     141: 0.44718165745163724,
     142: 0.49480776331438203,
     143: 0.4300665606377339,
     144: 0.3827166581199953,
     145: 0.4566916643445123,
     146: 0.41456866656069846,
     147: 0.44206750308919857,
     148: 0.39800640928871006,
     149: 0.4091276722070229,
     150: 0.41581736736359215,
     151: 0.4124453955411814,
     152: 0.4539030351117839,
     153: 0.44905184096328443,
     154: 0.44842671079024965,
     155: 0.45973337856945196,
     156: 0.40264981739707834,
     157: 0.4257090580409639,
     158: 0.42402789019437825,
     159: 0.41960901509031184,
     160: 0.5738477207559312,
     161: 0.41997373392133336,
     162: 0.44905184096328443,
     163: 0.44718165745163724,
     164: 0.4253343141518433,
     165: 0.46459594315047503,
     166: 0.5102215215168892,
     167: 0.4236560989710601,
     168: 0.4511482547866424,
     169: 0.47603919298176756,
     170: 0.45973337856945196,
     171: 0.4251471895085737,
     172: 0.46148976205968867,
     173: 0.4619309568608929,
     174: 0.44780331869925305,
     175: 0.3474863580557311,
     176: 0.3958867520495649,
     177: 0.4378611516778378,
     178: 0.3910803568405456,
     179: 0.43845715143057534,
     180: 0.45432983627314905,
     181: 0.42421403062027574,
     182: 0.3857722801409134,
     183: 0.503050266399265,
     184: 0.46148976205968867,
     185: 0.35830906998627665,
     186: 0.4108671606092636,
     187: 0.4358861352065801,
     188: 0.4294931385568836,
     189: 0.44491692530063903,
     190: 0.42968410927211564,
     191: 0.439054775898677,
     192: 0.3642516252367087,
     193: 0.3921913805815698,
     194: 0.41653429385904656,
     195: 0.40930095796399324,
     196: 0.40930095796399324,
     197: 0.4262724136537221,
     198: 0.4103437629524366,
     199: 0.4281610818577705,
     200: 0.40843599397843955,
     201: 0.4424723268099762,
     202: 0.3385983047487695,
     203: 0.41924492917700135,
     204: 0.41191797176171696,
     205: 0.41279776238914484,
     206: 0.43064151593270417,
     207: 0.42070507694949416,
     208: 0.42272946708354686,
     209: 0.45799031362700854,
     210: 0.4637042042960596,
     211: 0.49329227246196433,
     212: 0.48756789190362665,
     213: 0.38393307975883517,
     214: 0.4378611516778378,
     215: 0.46616476688518477,
     216: 0.4091276722070229,
     217: 0.4046731833136466,
     218: 0.4398541473613965,
     219: 0.39155573814950895,
     220: 0.3901330487496924,
     221: 0.4372667700239765,
     222: 0.44226982231258033,
     223: 0.4345141914356961,
     224: 0.39362914939021915,
     225: 0.4033220207650201,
     226: 0.4457378052366181,
     227: 0.38747376172934567,
     228: 0.4108671606092636,
     229: 0.4124453955411814,
     230: 0.42088831086802614,
     231: 0.41653429385904656,
     232: 0.45712372835997545,
     233: 0.44656171984888543,
     234: 0.43687141128073603,
     235: 0.39621138243254944,
     236: 0.43845715143057534,
     237: 0.4058628986782814,
     238: 0.4260844628540512,
     239: 0.36729743890269406,
     240: 0.39475472293831215,
     241: 0.3718197621211959,
     242: 0.42015633119695134,
     243: 0.4268372622583869,
     244: 0.4029856387627139,
     245: 0.430833509475251,
     246: 0.344882070575656,
     247: 0.3369454538887685,
     248: 0.38211133323566154,
     249: 0.5091462390690137,
     250: 0.41545982878460364,
     251: 0.415996367521734,
     252: 0.47254746295989636,
     253: 0.423099632991676,
     254: 0.4670660037472151,
     255: 0.4515698886696206,
     256: 0.4841480770305552,
     257: 0.3776317161989011,
     258: 0.4281610818577705,
     259: 0.39751524547634226,
     260: 0.3728239049972948,
     261: 0.4314105186397268,
     262: 0.40688823652757394,
     263: 0.44308095449472173,
     264: 0.41350430541420113,
     265: 0.3777793439222002,
     266: 0.39155573814950895,
     267: 0.3402674513214747,
     268: 0.4229144690385068,
     269: 0.4566916643445123,
     270: 0.4048427154390398,
     271: 0.44105867720355457,
     272: 0.42070507694949416,
     273: 0.38840818398432,
     274: 0.38887708722454245,
     275: 0.41924492917700135,
     276: 0.36646172231816004,
     277: 0.3920322765732203,
     278: 0.4438950674106514,
     279: 0.3998177748253985,
     280: 0.47697905318508793,
     281: 0.4394540981141374,
     282: 0.49103636267936385,
     283: 0.4960778037746345,
     284: 0.4298752498901192,
     285: 0.4599521950276002,
     286: 0.4223599483186137,
     287: 0.39899238718125024,
     288: 0.423099632991676,
     289: 0.3461173215447665,
     290: 0.46504309997737636,
     291: 0.4051822061857392,
     292: 0.4260844628540512,
     293: 0.3627475832406111,
     294: 0.4039964723047609,
     295: 0.4400544452427086,
     296: 0.4457378052366181,
     297: 0.3857722801409134,
     298: 0.3933087349422011,
     299: 0.3804565203751921,
     300: 0.45604509757101847,
     301: 0.48317978087649405,
     302: 0.415996367521734,
     303: 0.4702479619235952,
     304: 0.4210717044675329,
     305: 0.4268372622583869,
     306: 0.44553230140755556,
     307: 0.39459353276969705,
     308: 0.4366739998883814,
     309: 0.41104192333176864,
     310: 0.41942689312195663,
     311: 0.40064658447470486,
     312: 0.42347044774451714,
     313: 0.3595087655331057,
     314: 0.43237564284250024,
     315: 0.34414514307442595,
     316: 0.42199107500130484,
     317: 0.3785192172945508,
     318: 0.44614938215742755,
     319: 0.3813573645434049,
     320: 0.4260844628540512,
     321: 0.43179605082796607,
     322: 0.3792619944085511,
     323: 0.41906312304986476,
     324: 0.3896611136100758,
     325: 0.42199107500130484,
     326: 0.4364767668261012,
     327: 0.4298752498901192,
     328: 0.3800076923920519,
     329: 0.46549111837812523,
     330: 0.3588412780367575,
     331: 0.39686224301970763,
     332: 0.3555406776133142,
     333: 0.4873220180297469,
     334: 0.3498767421263534,
     335: 0.353977861447981,
     336: 0.4270258779288502,
     337: 0.42421403062027574,
     338: 0.43825830464988125,
     339: 0.4172536967845371,
     340: 0.4675179302143145,
     341: 0.35488783024347703,
     342: 0.439054775898677,
     343: 0.4133274430081215,
     344: 0.34811223406087466,
     345: 0.40281765808794834,
     346: 0.4212552579568387,
     347: 0.405352165164844,
     348: 0.32319717784380875,
     349: 0.40501238967015424,
     350: 0.38332390390836496,
     351: 0.442675016836,
     352: 0.4058628986782814,
     353: 0.42180687985726234,
     354: 0.3494971290245888,
     355: 0.4432842026389854,
     356: 0.4347096544098012,
     357: 0.43064151593270417,
     358: 0.3920322765732203,
     359: 0.3638401964431431,
     360: 0.3994872103154147,
     361: 0.4294931385568836,
     362: 0.45951477021064574,
     363: 0.43885538680880476,
     364: 0.3709633634368476,
     365: 0.4741705406049991,
     366: 0.4753367249153901,
     367: 0.45582998195895663,
     368: 0.4312180105992807,
     369: 0.382262484870644,
     370: 0.3752852667001895,
     371: 0.46348180419807583,
     372: 0.41743393596241385,
     373: 0.43687141128073603,
     374: 0.37572300223677607,
     375: 0.4451218617010539,
     376: 0.41906312304986476,
     377: 0.46594000084522086,
     378: 0.37484854994297445,
     379: 0.4262724136537221,
     380: 0.41456866656069846,
     381: 0.39899238718125024,
     382: 0.33706297933484064,
     383: 0.3319682451916826,
     384: 0.3698276164381891,
     385: 0.3998177748253985,
     386: 0.3533307355586794,
     387: 0.3933087349422011,
     388: 0.40637492083809423,
     389: 0.4236560989710601,
     390: 0.41226943760793006,
     391: 0.36996920434647323,
     392: 0.3854645240339003,
     393: 0.4505172782065212,
     394: 0.4011455216907381,
     395: 0.4272146603682529,
     396: 0.35153130656711096,
     397: 0.41492467228552515,
     398: 0.33117188545338866,
     399: 0.3896611136100758,
     400: 0.4321822726981163,
     401: 0.4294931385568836,
     402: 0.3786675398718605,
     403: 0.38654382470119525,
     404: 0.45432983627314905,
     405: 0.4836634443208148,
     406: 0.3970252924211126,
     407: 0.3749940092173023,
     408: 0.4170736131864428,
     409: 0.44842671079024965,
     410: 0.39491604485205883,
     411: 0.4632596173312503,
     412: 0.4262724136537221,
     413: 0.4070596300560186,
     414: 0.30619757976964135,
     415: 0.34451321274616326,
     416: 0.33461203661807065,
     417: 0.4424723268099762,
     418: 0.41350430541420113,
     419: 0.4709354589439513,
     420: 0.49103636267936385,
     421: 0.43490529331817646,
     422: 0.44635545577505226,
     423: 0.42911170592939074,
     424: 0.4880603847237313,
     425: 0.40637492083809423,
     426: 0.44226982231258033,
     427: 0.43766284499682434,
     428: 0.39459353276969705,
     429: 0.40315375959657407,
     430: 0.45009760677828975,
     431: 0.41191797176171696,
     432: 0.35991045130465105,
     433: 0.3877847358559342,
     434: 0.5159421045130743,
     435: 0.38887708722454245,
     436: 0.33004083393203143,
     437: 0.3877847358559342,
     438: 0.45009760677828975,
     439: 0.3368280103705082,
     440: 0.4505172782065212,
     441: 0.43766284499682434,
     442: 0.3728239049972948,
     443: 0.410169593273764,
     444: 0.4133274430081215,
     445: 0.4251471895085737,
     446: 0.40248211651519705,
     447: 0.3910803568405456,
     448: 0.41779488186467273,
     449: 0.2334201839983063,
     450: 0.45454353798353153,
     451: 0.34050724515609165,
     452: 0.3994872103154147,
     453: 0.42496022944282674,
     454: 0.39735179348395894,
     455: 0.40791876815237993,
     456: 0.3472366373528524,
     457: 0.41403580195072326,
     458: 0.442675016836,
     459: 0.43885538680880476,
     460: 0.4364767668261012,
     461: 0.3728239049972948,
     462: 0.41492467228552515,
     463: 0.2916871602031355,
     464: 0.4400544452427086,
     465: 0.38030679329121925,
     466: 0.4260844628540512,
     467: 0.4210717044675329,
     468: 0.40947439057330004,
     469: 0.37807494591274965,
     470: 0.39171445551397976,
     471: 0.40620410330096174,
     472: 0.3364761705268064,
     473: 0.47768638742115077,
     474: 0.4575566106784981,
     475: 0.3222272630053311,
     476: 0.357117354675901,
     477: 0.34377785903699326,
     478: 0.42402789019437825,
     479: 0.31539150187760706,
     480: 0.4354932680274845,
     481: 0.42347044774451714,
     482: 0.42421403062027574,
     483: 0.4279714622466732,
     484: 0.41403580195072326,
     485: 0.39899238718125024,
     486: 0.460829547807815,
     487: 0.3487403687307788,
     488: 0.34599339840780097,
     489: 0.42873095020097074,
     490: 0.4374647178601123,
     491: 0.3715338568831173,
     492: 0.42033908732187386,
     493: 0.4562604163139698,
     494: 0.4839056393354973,
     495: 0.45690759420945065,
     496: 0.40947439057330004,
     497: 0.38120692771321024,
     498: 0.45178100128704446,
     499: 0.39751524547634226,
     500: 0.3393116438739424,
     501: 0.38747376172934567,
     502: 0.35698543101329444,
     503: 0.3417113018928529,
     504: 0.41779488186467273,
     505: 0.3708210137194889,
     506: 0.4451218617010539,
     507: 0.33219648049260503,
     508: 0.4051822061857392,
     509: 0.4260844628540512,
     510: 0.3771895244937502,
     511: 0.40248211651519705,
     512: 0.374267839563512,
     513: 0.43845715143057534,
     514: 0.36729743890269406,
     515: 0.4188814745353221,
     516: 0.369262346867783,
     517: 0.3691212993708893,
     518: 0.4522038192573646,
     519: 0.3971884758540847,
     520: 0.43315085690407357,
     521: 0.38485048257785265,
     522: 0.31725527306401446,
     523: 0.374267839563512,
     524: 0.2562608225279735,
     525: 0.39459353276969705,
     526: 0.44676817464308277,
     527: 0.3870082345826945,
     528: 0.3718197621211959,
     529: 0.40147883745450275,
     530: 0.36590668752479666,
     531: 0.4272146603682529,
     532: 0.3783710108664793,
     533: 0.4968429623408679,
     534: 0.34377785903699326,
     535: 0.36085121798095143,
     536: 0.3860805280675142,
     537: 0.4244003345423751,
     538: 0.3419531357936971,
     539: 0.37645483511997974,
     540: 0.38669850410283635,
     541: 0.4106925464313591,
     542: 0.3718197621211959,
     543: 0.41617552185744533,
     544: 0.4124453955411814,
     545: 0.3797090615925297,
     546: 0.48149454995166324,
     547: 0.37325591415719894,
     548: 0.4277820105148243,
     549: 0.43865617873490154,
     550: 0.4216228454419669,
     551: 0.4072311680375002,
     552: 0.4229144690385068,
     553: 0.36562980013355584,
     554: 0.3718197621211959,
     555: 0.3827166581199953,
     556: 0.3488662677808621,
     557: 0.4055222667868183,
     558: 0.34086757028324094,
     559: 0.38423839433518414,
     560: 0.4277820105148243,
     561: 0.2916871602031355,
     562: 0.42199107500130484,
     563: 0.39866318554166175,
     564: 0.39965242421546243,
     565: 0.3395500919722375,
     566: 0.3482376799109867,
     567: 0.35423737600915983,
     568: 0.40382764803718685,
     569: 0.3786675398718605,
     570: 0.34122865881108333,
     571: 0.385157258570342,
     572: 0.41743393596241385,
     573: 0.3983345266912564,
     574: 0.3347279396442633,
     575: 0.3519153538794567,
     576: 0.41671391192453133,
     577: 0.38825213409119647,
     578: 0.2782492259582459,
     579: 0.37412294299380106,
     580: 0.0,
     581: 0.41297417168931116,
     582: 0.41315073183111933,
     583: 0.276497728684689,
     584: 0.35567153542620095,
     585: 0.36452642842436367,
     586: 0.3944324741848931,
     587: 0.35358930177569997,
     588: 0.38393307975883517,
     589: 0.405352165164844,
     590: 0.3933087349422011,
     591: 0.39187330160299594,
     592: 0.39362914939021915,
     593: 0.41510290453307047,
     594: 0.37167675452038,
     595: 0.27760975632088136,
     596: 0.35449727136940135,
     597: 0.38825213409119647,
     598: 0.3649394115381375,
     599: 0.3588412780367575,
     600: 0.35540991605479516,
     601: 0.44166341944834925,
     602: 0.3876291864231801,
     603: 0.2334201839983063,
     604: 0.3641143789574182,
     605: 0.26533760619247343,
     606: 0.3118294810432359,
     607: 0.41104192333176864,
     608: 0.36535333147560983,
     609: 0.34207418115150023,
     610: 0.3976788319971144,
     611: 0.4268372622583869,
     612: 0.43568961305364656,
     613: 0.3407473772048618,
     614: 0.329590573585603,
     615: 0.4505172782065212,
     616: 0.3492445109334977,
     617: 0.32363012784761824,
     618: 0.3417113018928529,
     619: 0.3595087655331057,
     620: 0.34811223406087466,
     621: 0.3988277184288024,
     622: 0.29715853682441207,
     623: 0.3731117999046286,
     624: 0.3907640767298779,
     625: 0.3385983047487695,
     626: 0.328470279317807,
     627: 0.39735179348395894,
     628: 0.32625238411647134,
     629: 0.3257025823232181,
     630: 0.318090704987817,
     631: 0.31498030044100006,
     632: 0.29936789397552294,
     633: 0.0,
     634: 0.25277519271592674,
     635: 0.34038730600668826,
     636: 0.3283586686214707,
     637: 0.32126315217852,
     638: 0.38638926899359777,
     639: 0.3402674513214747,
     640: 0.3663228058199348,
     641: 0.40382764803718685,
     642: 0.39171445551397976,
     643: 0.3729677968942447,
     644: 0.39491604485205883,
     645: 0.31840512743096805,
     646: 0.3374160480981104,
     647: 0.3868533073470729,
     648: 0.0,
     649: 0.3709633634368476,
     650: 0.31611369373666603,
     651: 0.39621138243254944,
     652: 0.3532015942079635,
     653: 0.0,
     654: 0.42458680217618106,
     655: 0.40569251123131317,
     656: 0.3391925453678442,
     657: 0.31172889088806066,
     658: 0.0,
     659: 0.3160103210441426,
     660: 0.0,
     661: 0.41510290453307047,
     662: 0.39427154702284295,
     663: 0.39282909014349104,
     664: 0.3827166581199953,
     665: 0.40131211036253656,
     666: 0.4055222667868183,
     667: 0.41174246346526977,
     668: 0.2969758948226761,
     669: 0.3577784382647124,
     670: 0.0,
     671: 0.3907640767298779,
     672: 0.39187330160299594,
     673: 0.3118294810432359,
     674: 0.3341492260556667,
     675: 0.0,
     676: 0.3657681914280803,
     677: 0.3428022567410387,
     678: 0.3493707743141678,
     679: 0.3904483077789851,
     680: 0.2915111800159843,
     681: 0.41492467228552515,
     682: 0.31142750942732456,
     683: 0.3472366373528524,
     684: 0.0,
     685: 0.41121683478850557,
     686: 0.3818093882864433,
     687: 0.3709633634368476,
     688: 0.2923047676203836,
     689: 0.3051340580211519,
     690: 0.4143908926899606,
     691: 0.0,
     692: 0.3075619229003781,
     693: 0.367856704131324,
     694: 0.35410757118101427,
     695: 0.40620410330096174,
     696: 0.33943082604600916,
     697: 0.34711191154920545,
     698: 0.376748367155161,
     699: 0.35423737600915983,
     700: 0.3563272720328127,
     701: 0.2916871602031355,
     702: 0.3425592207561106,
     703: 0.0,
     704: 0.34159051316825306,
     705: 0.3701109007096852,
     706: 0.3686988026527997,
     707: 0.38747376172934567,
     708: 0.38531083004505107,
     709: 0.3144678040198464,
     710: 0.37210610772159725,
     711: 0.0,
     712: 0.3312854171247816,
     713: 0.3253735898158209,
     714: 0.40081275891870094,
     715: 0.3497501128313384,
     716: 0.3319682451916826,
     717: 0.3898182984078209,
     718: 0.3496235751638886,
     719: 0.3770423572973032,
     720: 0.36562980013355584,
     721: 0.3251546304687039,
     722: 0.33105843157005416,
     723: 0.3588412780367575,
     724: 0.3671578882040228,
     725: 0.3573814947311346,
     726: 0.410169593273764,
     727: 0.34911833878359394,
     728: 0.3738334861713687,
     729: 0.3552792506444809,
     730: 0.41368131924357365,
     731: 0.0,
     732: 0.0,
     733: 0.3641143789574182,
     734: 0.38211133323566154,
     735: 0.3234135079494605,
     736: 0.3904483077789851,
     737: 0.30436521630015373,
     738: 0.39621138243254944,
     739: 0.3637032599747791,
     740: 0.37239289470250025,
     741: 0.329590573585603,
     742: 0.34390019991209536,
     743: 0.28002305469515737,
     744: 0.0,
     745: 0.3597764563488414,
     746: 0.0,
     747: 0.4277820105148243,
     748: 0.4274036097978718,
     749: 0.36085121798095143,
     750: 0.2902852393370346,
     751: 0.35924147277062757,
     752: 0.37572300223677607,
     753: 0.3708210137194889,
     754: 0.39187330160299594,
     755: 0.3283586686214707,
     756: 0.4018127075896,
     757: 0.31694311635060285,
     758: 0.37470320347149594,
     759: 0.3506384476607359,
     760: 0.33094505539485897,
     761: 0.30883974488750016,
     762: 0.2683586675237401,
     763: 0.3319682451916826,
     764: 0.37733680661967517,
     765: 0.37748420380976094,
     766: 0.32254992047830044,
     767: 0.3801571840098301,
     768: 0.35410757118101427,
     769: 0.35165922916775405,
     770: 0.29507162190930936,
     771: 0.37911320586621733,
     772: 0.0,
     773: 0.3240642393537854,
     774: 0.3132445905196072,
     775: 0.2979832136148591,
     776: 0.4009790712668,
     777: 0.3909221528126975,
     778: 0.37881597873500117,
     779: 0.33765183848811603,
     780: 0.3234135079494605,
     781: 0.30854392137707154,
     782: 0.3619324201322053,
     783: 0.36562980013355584,
     784: 0.31477510154820454,
     785: 0.34586956397744745,
     786: 0.3901330487496924,
     787: 0.32592228052377337,
     788: 0.27905271780334623,
     789: 0.3734001397809073,
     790: 0.2904597420357644,
     791: 0.3380061426208423,
     792: 0.34475903023652804,
     793: 0.3221198539176627,
     794: 0.27492448414025267,
     795: 0.3050377404523321,
     796: 0.3532015942079635,
     797: 0.33253942248898416,
     798: 0.0,
     799: 0.3843912337919602,
     800: 0.35242872419875565,
     801: 0.32691460140493506,
     802: 0.35307254722432885,
     803: 0.35423737600915983,
     804: 0.3467382711707887,
     805: 0.3335725100976832,
     806: 0.3268040452326642,
     807: 0.35645870961010256,
     808: 0.0,
     809: 0.4279714622466732,
     810: 0.3876291864231801,
     811: 0.3666007442158528,
     812: 0.4360828347260776,
     813: 0.3734001397809073,
     814: 0.3350761309823121,
     815: 0.37354447690490455,
     816: 0.3663228058199348,
     817: 0.3219052504173844,
     818: 0.40947439057330004,
     819: 0.3197748384358002,
     820: 0.4851202619241908,
     821: 0.3668791046898208,
     822: 0.3912386889688211,
     823: 0.3511480965672195,
     824: 0.3034095955268409,
     825: 0.3102277886847474,
     826: 0.38872066039943204,
     827: 0.3062946313004717,
     828: 0.41869998342850434,
     829: 0.3272467191848926,
     830: 0.28397283624830677,
     831: 0.29816709711600986,
     832: 0.38825213409119647,
     833: 0.3789645340207797,
     834: 0.34159051316825306,
     835: 0.34911833878359394,
     836: 0.29955349093397027,
     837: 0.34159051316825306,
     838: 0.3033143633876297,
     839: 0.3248267434463825,
     840: 0.4039964723047609,
     841: 0.32287322477547215,
     842: 0.34134919171776335,
     843: 0.3159070159375574,
     844: 0.3450051987693638,
     845: 0.382262484870644,
     846: 0.20889744093233636,
     847: 0.37733680661967517,
     848: 0.33219648049260503,
     849: 0.3476113531485569,
     850: 0.33015359130611144,
     851: 0.3451284149117815,
     852: 0.3190358407900257,
     853: 0.34599339840780097,
     854: 0.3752852667001895,
     855: 0.3617969156694077,
     856: 0.440254925627785,
     857: 0.3792619944085511,
     858: 0.3064889190463013,
     859: 0.44759590632375545,
     860: 0.35567153542620095,
     861: 0.27570886212638746,
     862: 0.2796179287479711,
     863: 0.2989049062025945,
     864: 0.31477510154820454,
     865: 0.3374160480981104,
     866: 0.34086757028324094,
     867: 0.344882070575656,
     868: 0.3407473772048618,
     869: 0.31725527306401446,
     870: 0.3205172675797639,
     871: 0.29248170755235714,
     872: 0.36288380088358546,
     873: 0.30455706326914217,
     874: 0.3205172675797639,
     875: 0.31314308546759173,
     876: 0.2840563085693674,
     877: 0.3901330487496924,
     878: 0.31704710031266015,
     879: 0.29426296033891236,
     880: 0.4039964723047609,
     881: 0.31673535291805577,
     882: 0.3686988026527997,
     883: 0.3418321760710959,
     884: 0.2972499420956592,
     885: 0.3391925453678442,
     886: 0.3500034631484926,
     887: 0.3162171340814751,
     888: 0.2972499420956592,
     889: 0.33391830053662336,
     890: 0.34475903023652804,
     891: 0.32439058803390003,
     892: 0.41174246346526977,
     893: 0.31914120269253243,
     894: 0.29899738915624635,
     895: 0.3401476810112594,
     896: 0.4205220025034761,
     897: 0.30532687575133904,
     898: 0.30726854109792945,
     899: 0.3139569726293009,
     900: 0.3955626531940189,
     901: 0.31092649992052385,
     902: 0.3319682451916826,
     903: 0.30864246622580266,
     904: 0.2999253760872092,
     905: 0.33978887544057246,
     906: 0.37881597873500117,
     907: 0.3425592207561106,
     908: 0.3670184435066419,
     909: 0.3709633634368476,
     910: 0.3137531044652559,
     911: 0.3329977814448615,
     912: 0.3222272630053311,
     913: 0.3663228058199348,
     914: 0.37412294299380106,
     915: 0.28786403388531073,
     916: 0.2334201839983063,
     917: 0.3637032599747791,
     918: 0.3857722801409134,
     919: 0.35281473594486606,
     920: 0.2988124804431008,
     921: 0.3519153538794567,
     922: 0.3804565203751921,
     923: 0.3062946313004717,
     924: 0.3617969156694077,
     925: 0.3572493758791083,
     926: 0.3617969156694077,
     927: 0.32647282491655005,
     928: 0.3283586686214707,
     929: 0.32769059401593353,
     930: 0.34911833878359394,
     931: 0.36729743890269406,
     932: 0.4723165013455465,
     933: 0.3418321760710959,
     934: 0.3381244092907586,
     935: 0.3120308562328021,
     936: 0.3873184616244441,
     937: 0.3466139030677862,
     938: 0.2975244956136047,
     939: 0.32126315217852,
     940: 0.33671064869442097,
     941: 0.30619757976964135,
     942: 0.3042693834234849,
     943: 0.3038866546393044,
     944: 0.29779955678058184,
     945: 0.32233474374682725,
     946: 0.28615918322564055,
     947: 0.3381244092907586,
     948: 0.28372271337433586,
     949: 0.3813573645434049,
     950: 0.43925434625135823,
     951: 0.3293659038012911,
     952: 0.3344962138293486,
     953: 0.36329306832819097,
     954: 0.36356642654363736,
     955: 0.3428022567410387,
     956: 0.36342969603346675,
     957: 0.32276538468703675,
     958: 0.30571324319930027,
     959: 0.3160103210441426,
     960: 0.3160103210441426,
     961: 0.3160103210441426,
     962: 0.2953421643499352,
     963: 0.42143897154513216,
     964: 0.3292536837318528,
     965: 0.2777693480175304,
     966: 0.3327684441298169,
     967: 0.3285819659139708,
     968: 0.31487766756369767,
     969: 0.31673535291805577,
     970: 0.3555406776133142,
     971: 0.44308095449472173,
     972: 0.3041736108759799,
     973: 0.30532687575133904,
     974: 0.35281473594486606,
     975: 0.30532687575133904,
     976: 0.3617969156694077,
     977: 0.3139569726293009,
     978: 0.3316264796681497,
     979: 0.34353343823426524,
     980: 0.37239289470250025,
     981: 0.3617969156694077,
     982: 0.31092649992052385,
     983: 0.34122865881108333,
     984: 0.3327684441298169,
     985: 0.2937263105632183,
     986: 0.3255928442563976,
     987: 0.3558024895997747,
     988: 0.3200926007793932,
     989: 0.31767243976100856,
     990: 0.3729677968942447,
     991: 0.322012516412192,
     992: 0.3329977814448615,
     993: 0.29274751946470406,
     994: 0.3205172675797639,
     995: 0.24772098481235275,
     996: 0.30057840178942086,
     997: 0.308052139545103,
     998: 0.3137531044652559,
     999: 0.31172889088806066,
     ...}




```python
# Add closeness centrality score for each node
# As G is not connected (has 20 connected components), use the normalised betweenness score
dfG['closeness'] = pd.Series(nx.closeness_centrality(G, normalized=True))
```


```python
print('Average closeness of management is {0}\nAverage closeness of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].closeness.mean(), dfG[dfG.ManagementSalary==0.0].closeness.mean()))
```

    Average closeness of management is 0.44643327326051496
    Average closeness of non-management is 0.36692580135088176
    


```python
# Add betweenness centrality score for each node
# Approximate using a sub-sample of nodes due to having 1005 nodes
# Include nodes being measured in score based on their likelihood of having short paths to other 
# managers and staff, and non-managers having less direct connections
# As G is not connected (has 20 connected components), use the normalised betweenness score
dfG['btwn'] = pd.Series(nx.betweenness_centrality(G, k=100, normalized=True, endpoints=True, seed=1))
dfG['btwnEndFalse'] = pd.Series(nx.betweenness_centrality(G, k=100, normalized=True, endpoints=False, seed=1))
```


```python
print('Average betweenness of management is {0}\nAverage betweeness of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].btwn.mean(), dfG[dfG.ManagementSalary==0.0].btwn.mean()))
print('Average betweenness (endpoint=False) of management is {0}\nAverage betweeness (endpoint=False) of non-management is {1}'.format(
dfG[dfG.ManagementSalary==1.0].btwnEndFalse.mean(), dfG[dfG.ManagementSalary==0.0].btwnEndFalse.mean()))
```

    Average betweenness of management is 0.008919185837671487
    Average betweeness of non-management is 0.0026407060639611884
    Average betweenness (endpoint=False) of management is 0.006930897463649289
    Average betweeness (endpoint=Falseof non-management is 0.0007059674041253489
    


```python
dfG.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Department</th>
      <th>ManagementSalary</th>
      <th>degree</th>
      <th>LCC</th>
      <th>closeness</th>
      <th>btwn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>44</td>
      <td>0.276423</td>
      <td>0.421991</td>
      <td>0.002261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>NaN</td>
      <td>52</td>
      <td>0.265306</td>
      <td>0.422360</td>
      <td>0.001974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>NaN</td>
      <td>95</td>
      <td>0.297803</td>
      <td>0.461490</td>
      <td>0.019063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1.0</td>
      <td>71</td>
      <td>0.384910</td>
      <td>0.441663</td>
      <td>0.004131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>1.0</td>
      <td>96</td>
      <td>0.318691</td>
      <td>0.462152</td>
      <td>0.015059</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler
```


```python
StandardScaler?
```


```python
from sklearn.model_selection import GridSearchCV
```


```python
GridSearchCV?
```


```python

```


```python

```


```python
from sklearn.model_selection import cross_val_score
```


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
PolynomialFeatures?
```

# Part 2B (Future connections)


```python
pd.read_csv?
```


```python
future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
future_connections.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Future Connection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(6, 840)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(4, 197)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(620, 979)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(519, 872)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(382, 423)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(97, 226)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>(349, 905)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(429, 860)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(309, 989)</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>(468, 880)</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Load a fresh version of the graph
Ge = nx.read_gpickle('email_prediction.txt')
print(nx.info(Ge))
```

    Name: 
    Type: Graph
    Number of nodes: 1005
    Number of edges: 16706
    Average degree:  33.2458
    


```python
len(future_connections)
```




    488446




```python
future_connections['Future Connection'].value_counts(dropna=False)
```




     0.0    337002
    NaN     122112
     1.0     29332
    Name: Future Connection, dtype: int64




```python
future_connections['commNeigh'] = future_connections.index.map(lambda x: 
                             len(list(nx.common_neighbors(Ge, x[0], x[1]))))
```


```python
future_connections['jaccard'] = [x[2] for x in list(nx.jaccard_coefficient(Ge, future_connections.index))]
```


```python
future_connections['prefAtt'] = [x[2] for x in list(nx.preferential_attachment(Ge, future_connections.index))]
```


```python
future_connections['communityCN'] = [x[2] for x in list(
    nx.cn_soundarajan_hopcroft(Ge, future_connections.index, community='Department'))]
```


```python
future_connections['communityRA'] = [x[2] for x in list(
    nx.ra_index_soundarajan_hopcroft(Ge, future_connections.index, community='Department'))]
```


```python
future_connections['raIndex'] = [x[2] for x in list(
    nx.resource_allocation_index(Ge, future_connections.index))]
```


```python
future_connections['aaIndex'] = [x[2] for x in list(
    nx.adamic_adar_index(Ge, future_connections.index))]
```


```python
nx.ra_index_soundarajan_hopcroft?
```


```python
from sklearn.linear_model import LogisticRegression
LogisticRegression?
```


```python
future_connections.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Future Connection</th>
      <th>commNeigh</th>
      <th>jaccard</th>
      <th>prefAtt</th>
      <th>communityCN</th>
      <th>aaIndex</th>
      <th>raIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(6, 840)</th>
      <td>0.0</td>
      <td>9</td>
      <td>0.073770</td>
      <td>2070</td>
      <td>9</td>
      <td>2.110314</td>
      <td>0.136721</td>
    </tr>
    <tr>
      <th>(4, 197)</th>
      <td>0.0</td>
      <td>2</td>
      <td>0.015504</td>
      <td>3552</td>
      <td>2</td>
      <td>0.363528</td>
      <td>0.008437</td>
    </tr>
    <tr>
      <th>(620, 979)</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>28</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(519, 872)</th>
      <td>0.0</td>
      <td>2</td>
      <td>0.060606</td>
      <td>299</td>
      <td>2</td>
      <td>0.507553</td>
      <td>0.039726</td>
    </tr>
    <tr>
      <th>(382, 423)</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>205</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(97, 226)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.048193</td>
      <td>1575</td>
      <td>4</td>
      <td>0.843588</td>
      <td>0.036296</td>
    </tr>
    <tr>
      <th>(349, 905)</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>240</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(429, 860)</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>816</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(309, 989)</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>184</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>(468, 880)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.019608</td>
      <td>672</td>
      <td>1</td>
      <td>0.170960</td>
      <td>0.002882</td>
    </tr>
  </tbody>
</table>
</div>




```python
future_connections[future_connections.commNeigh != future_connections.communityCN]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Future Connection</th>
      <th>commNeigh</th>
      <th>jaccard</th>
      <th>prefAtt</th>
      <th>communityCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(707, 922)</th>
      <td>1.0</td>
      <td>5</td>
      <td>0.084746</td>
      <td>1000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>(530, 661)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.080000</td>
      <td>450</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(360, 362)</th>
      <td>1.0</td>
      <td>13</td>
      <td>0.152941</td>
      <td>2100</td>
      <td>23</td>
    </tr>
    <tr>
      <th>(93, 198)</th>
      <td>1.0</td>
      <td>14</td>
      <td>0.128440</td>
      <td>3306</td>
      <td>20</td>
    </tr>
    <tr>
      <th>(551, 911)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.045455</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(115, 724)</th>
      <td>0.0</td>
      <td>2</td>
      <td>0.015385</td>
      <td>262</td>
      <td>3</td>
    </tr>
    <tr>
      <th>(171, 725)</th>
      <td>1.0</td>
      <td>11</td>
      <td>0.239130</td>
      <td>585</td>
      <td>18</td>
    </tr>
    <tr>
      <th>(511, 656)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.030303</td>
      <td>264</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(36, 39)</th>
      <td>1.0</td>
      <td>18</td>
      <td>0.295082</td>
      <td>1620</td>
      <td>28</td>
    </tr>
    <tr>
      <th>(145, 954)</th>
      <td>1.0</td>
      <td>5</td>
      <td>0.069444</td>
      <td>497</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(324, 358)</th>
      <td>1.0</td>
      <td>7</td>
      <td>0.106061</td>
      <td>1045</td>
      <td>13</td>
    </tr>
    <tr>
      <th>(231, 352)</th>
      <td>1.0</td>
      <td>14</td>
      <td>0.184211</td>
      <td>1792</td>
      <td>23</td>
    </tr>
    <tr>
      <th>(246, 674)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.121212</td>
      <td>380</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(474, 713)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.012048</td>
      <td>246</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(555, 570)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.125000</td>
      <td>336</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(441, 765)</th>
      <td>1.0</td>
      <td>9</td>
      <td>0.136364</td>
      <td>1100</td>
      <td>17</td>
    </tr>
    <tr>
      <th>(707, 754)</th>
      <td>1.0</td>
      <td>11</td>
      <td>0.192982</td>
      <td>1120</td>
      <td>21</td>
    </tr>
    <tr>
      <th>(356, 652)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.075472</td>
      <td>357</td>
      <td>7</td>
    </tr>
    <tr>
      <th>(172, 789)</th>
      <td>1.0</td>
      <td>6</td>
      <td>0.062500</td>
      <td>1012</td>
      <td>12</td>
    </tr>
    <tr>
      <th>(406, 674)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.090909</td>
      <td>600</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(403, 523)</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.117647</td>
      <td>380</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(206, 270)</th>
      <td>1.0</td>
      <td>5</td>
      <td>0.071429</td>
      <td>915</td>
      <td>9</td>
    </tr>
    <tr>
      <th>(344, 668)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.055556</td>
      <td>19</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(360, 565)</th>
      <td>1.0</td>
      <td>3</td>
      <td>0.078947</td>
      <td>360</td>
      <td>6</td>
    </tr>
    <tr>
      <th>(299, 972)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.040000</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(466, 699)</th>
      <td>1.0</td>
      <td>16</td>
      <td>0.262295</td>
      <td>1378</td>
      <td>31</td>
    </tr>
    <tr>
      <th>(142, 372)</th>
      <td>1.0</td>
      <td>17</td>
      <td>0.111111</td>
      <td>4371</td>
      <td>27</td>
    </tr>
    <tr>
      <th>(411, 737)</th>
      <td>0.0</td>
      <td>1</td>
      <td>0.009434</td>
      <td>318</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(505, 805)</th>
      <td>1.0</td>
      <td>2</td>
      <td>0.133333</td>
      <td>52</td>
      <td>3</td>
    </tr>
    <tr>
      <th>(440, 910)</th>
      <td>1.0</td>
      <td>1</td>
      <td>0.014286</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>(157, 814)</th>
      <td>NaN</td>
      <td>3</td>
      <td>0.037500</td>
      <td>395</td>
      <td>5</td>
    </tr>
    <tr>
      <th>(119, 847)</th>
      <td>NaN</td>
      <td>5</td>
      <td>0.135135</td>
      <td>406</td>
      <td>6</td>
    </tr>
    <tr>
      <th>(19, 496)</th>
      <td>NaN</td>
      <td>11</td>
      <td>0.130952</td>
      <td>1800</td>
      <td>17</td>
    </tr>
    <tr>
      <th>(674, 700)</th>
      <td>NaN</td>
      <td>7</td>
      <td>0.166667</td>
      <td>600</td>
      <td>14</td>
    </tr>
    <tr>
      <th>(504, 700)</th>
      <td>NaN</td>
      <td>11</td>
      <td>0.177419</td>
      <td>1320</td>
      <td>21</td>
    </tr>
    <tr>
      <th>(182, 398)</th>
      <td>NaN</td>
      <td>4</td>
      <td>0.086957</td>
      <td>188</td>
      <td>7</td>
    </tr>
    <tr>
      <th>(149, 851)</th>
      <td>NaN</td>
      <td>6</td>
      <td>0.157895</td>
      <td>480</td>
      <td>11</td>
    </tr>
    <tr>
      <th>(291, 706)</th>
      <td>NaN</td>
      <td>3</td>
      <td>0.081081</td>
      <td>348</td>
      <td>6</td>
    </tr>
    <tr>
      <th>(474, 549)</th>
      <td>NaN</td>
      <td>10</td>
      <td>0.066225</td>
      <td>6642</td>
      <td>14</td>
    </tr>
    <tr>
      <th>(632, 869)</th>
      <td>NaN</td>
      <td>2</td>
      <td>0.250000</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>(65, 611)</th>
      <td>NaN</td>
      <td>9</td>
      <td>0.084112</td>
      <td>2366</td>
      <td>15</td>
    </tr>
    <tr>
      <th>(409, 651)</th>
      <td>NaN</td>
      <td>7</td>
      <td>0.072917</td>
      <td>1634</td>
      <td>10</td>
    </tr>
    <tr>
      <th>(351, 656)</th>
      <td>NaN</td>
      <td>5</td>
      <td>0.074627</td>
      <td>682</td>
      <td>9</td>
    </tr>
    <tr>
      <th>(227, 468)</th>
      <td>NaN</td>
      <td>4</td>
      <td>0.111111</td>
      <td>420</td>
      <td>5</td>
    </tr>
    <tr>
      <th>(542, 789)</th>
      <td>NaN</td>
      <td>5</td>
      <td>0.208333</td>
      <td>198</td>
      <td>10</td>
    </tr>
    <tr>
      <th>(367, 976)</th>
      <td>NaN</td>
      <td>9</td>
      <td>0.090000</td>
      <td>2268</td>
      <td>18</td>
    </tr>
    <tr>
      <th>(540, 678)</th>
      <td>NaN</td>
      <td>3</td>
      <td>0.142857</td>
      <td>144</td>
      <td>6</td>
    </tr>
    <tr>
      <th>(219, 224)</th>
      <td>NaN</td>
      <td>4</td>
      <td>0.142857</td>
      <td>270</td>
      <td>8</td>
    </tr>
    <tr>
      <th>(184, 778)</th>
      <td>NaN</td>
      <td>3</td>
      <td>0.037037</td>
      <td>400</td>
      <td>4</td>
    </tr>
    <tr>
      <th>(207, 419)</th>
      <td>NaN</td>
      <td>7</td>
      <td>0.057377</td>
      <td>3030</td>
      <td>11</td>
    </tr>
    <tr>
      <th>(277, 345)</th>
      <td>NaN</td>
      <td>6</td>
      <td>0.139535</td>
      <td>644</td>
      <td>11</td>
    </tr>
    <tr>
      <th>(473, 958)</th>
      <td>NaN</td>
      <td>2</td>
      <td>0.018182</td>
      <td>330</td>
      <td>4</td>
    </tr>
    <tr>
      <th>(95, 572)</th>
      <td>NaN</td>
      <td>8</td>
      <td>0.125000</td>
      <td>1248</td>
      <td>11</td>
    </tr>
    <tr>
      <th>(200, 1000)</th>
      <td>NaN</td>
      <td>2</td>
      <td>0.051282</td>
      <td>216</td>
      <td>3</td>
    </tr>
    <tr>
      <th>(721, 878)</th>
      <td>NaN</td>
      <td>1</td>
      <td>0.111111</td>
      <td>28</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(566, 699)</th>
      <td>NaN</td>
      <td>7</td>
      <td>0.175000</td>
      <td>572</td>
      <td>13</td>
    </tr>
    <tr>
      <th>(151, 895)</th>
      <td>NaN</td>
      <td>1</td>
      <td>0.021277</td>
      <td>138</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(440, 817)</th>
      <td>NaN</td>
      <td>3</td>
      <td>0.041096</td>
      <td>497</td>
      <td>6</td>
    </tr>
    <tr>
      <th>(467, 969)</th>
      <td>NaN</td>
      <td>1</td>
      <td>0.040000</td>
      <td>48</td>
      <td>2</td>
    </tr>
    <tr>
      <th>(149, 214)</th>
      <td>NaN</td>
      <td>11</td>
      <td>0.157143</td>
      <td>1590</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>10703 rows × 5 columns</p>
</div>




```python
dfG = pd.DataFrame(index=G.nodes())

dfG['Department'] = pd.Series(nx.get_node_attributes(G, 'Department'))
dfG['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))

```


```python
dfG.Department.max()
```




    41




```python
pd.DataFrame.corr?
```


```python
fc_samples = future_connections[future_connections['Future Connection'].notnull()]
fc_pred = future_connections[future_connections['Future Connection'].isnull()]
```


```python
pd.DataFrame.corr(fc_samples)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Future Connection</th>
      <th>commNeigh</th>
      <th>jaccard</th>
      <th>prefAtt</th>
      <th>communityCN</th>
      <th>aaIndex</th>
      <th>raIndex</th>
      <th>communityRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Future Connection</th>
      <td>1.000000</td>
      <td>0.651724</td>
      <td>0.603526</td>
      <td>0.476706</td>
      <td>0.676138</td>
      <td>0.652264</td>
      <td>0.614138</td>
      <td>0.307982</td>
    </tr>
    <tr>
      <th>commNeigh</th>
      <td>0.651724</td>
      <td>1.000000</td>
      <td>0.790935</td>
      <td>0.828457</td>
      <td>0.981878</td>
      <td>0.996303</td>
      <td>0.920304</td>
      <td>0.201959</td>
    </tr>
    <tr>
      <th>jaccard</th>
      <td>0.603526</td>
      <td>0.790935</td>
      <td>1.000000</td>
      <td>0.490644</td>
      <td>0.824241</td>
      <td>0.789606</td>
      <td>0.738698</td>
      <td>0.394345</td>
    </tr>
    <tr>
      <th>prefAtt</th>
      <td>0.476706</td>
      <td>0.828457</td>
      <td>0.490644</td>
      <td>1.000000</td>
      <td>0.784192</td>
      <td>0.824596</td>
      <td>0.758970</td>
      <td>0.023595</td>
    </tr>
    <tr>
      <th>communityCN</th>
      <td>0.676138</td>
      <td>0.981878</td>
      <td>0.824241</td>
      <td>0.784192</td>
      <td>1.000000</td>
      <td>0.984372</td>
      <td>0.931142</td>
      <td>0.375450</td>
    </tr>
    <tr>
      <th>aaIndex</th>
      <td>0.652264</td>
      <td>0.996303</td>
      <td>0.789606</td>
      <td>0.824596</td>
      <td>0.984372</td>
      <td>1.000000</td>
      <td>0.949795</td>
      <td>0.236065</td>
    </tr>
    <tr>
      <th>raIndex</th>
      <td>0.614138</td>
      <td>0.920304</td>
      <td>0.738698</td>
      <td>0.758970</td>
      <td>0.931142</td>
      <td>0.949795</td>
      <td>1.000000</td>
      <td>0.349416</td>
    </tr>
    <tr>
      <th>communityRA</th>
      <td>0.307982</td>
      <td>0.201959</td>
      <td>0.394345</td>
      <td>0.023595</td>
      <td>0.375450</td>
      <td>0.236065</td>
      <td>0.349416</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
nx.cn_soundarajan_hopcroft?
```


```python
nx.set_node_attributes(Ge, 'community')
```


```python
depts = nx.get_node_attributes(Ge, 'Department')
```


```python

```


```python
G
```




    <networkx.classes.graph.Graph at 0x21e69a81240>




```python

```

# Part 1


```python
P1_Graphs = pickle.load(open('A4_graphs','rb'))
P1_Graphs
```




    [<networkx.classes.graph.Graph at 0x273a72dad68>,
     <networkx.classes.graph.Graph at 0x273a4a6d400>,
     <networkx.classes.graph.Graph at 0x273a4a6de80>,
     <networkx.classes.graph.Graph at 0x273a362a358>,
     <networkx.classes.graph.Graph at 0x273a6514ac8>]




```python
for i in range(len(P1_Graphs)):
    graph = P1_Graphs[i]
    print('Graph {0} has {1} nodes and {2} edges'.format(i, len(graph.nodes()), len(graph.edges())))
    print(nx.info(graph))
    print()
```

    Graph 0 has 1000 nodes and 1996 edges
    Name: barabasi_albert_graph(1000,2)
    Type: Graph
    Number of nodes: 1000
    Number of edges: 1996
    Average degree:   3.9920
    
    Graph 1 has 1000 nodes and 5000 edges
    Name: watts_strogatz_graph(1000,10,0.05)
    Type: Graph
    Number of nodes: 1000
    Number of edges: 5000
    Average degree:  10.0000
    
    Graph 2 has 750 nodes and 1500 edges
    Name: watts_strogatz_graph(750,5,0.075)
    Type: Graph
    Number of nodes: 750
    Number of edges: 1500
    Average degree:   4.0000
    
    Graph 3 has 750 nodes and 2984 edges
    Name: barabasi_albert_graph(750,4)
    Type: Graph
    Number of nodes: 750
    Number of edges: 2984
    Average degree:   7.9573
    
    Graph 4 has 750 nodes and 1500 edges
    Name: watts_strogatz_graph(750,4,1)
    Type: Graph
    Number of nodes: 750
    Number of edges: 1500
    Average degree:   4.0000
    
    


```python
nx.info?
```


```python
nx.info(P1_Graphs[0])
```




    'Name: barabasi_albert_graph(1000,2)\nType: Graph\nNumber of nodes: 1000\nNumber of edges: 1996\nAverage degree:   3.9920'



## Plot the degree distribution for each graph


```python
G = P1_Graphs[4]
```


```python
degrees = G.degree() 
degree_values= sorted(set(degrees.values())) 
histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values] 
plt.bar(degree_values,histogram) 
plt.xlabel('Degree') 
plt.ylabel('Fraction of Nodes') 
plt.show()

```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="640">



```python

```
