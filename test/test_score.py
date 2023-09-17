import json
import re
import numpy as np

import sys
sys.path.insert(0, 'src')
from app.score import score

with open('src/pose_data.json') as fp:
    data = json.load(fp)
    fp.close()

labeled = {}
for img in data:
    filename = img['file']
    gps = re.match(r'.*/(?P<name>\w*)_(?P<pose>\w*).jpg', filename).groupdict()
    labeled[(gps['name'], gps['pose'])] = np.array(img['output'])

scores = {}
COMPS = [('Jonathan', 'I', 'Jonathan' ,'I'), 
         
         ('Alex', 'I', 'Jonathan' ,'I'), 
         ('Jonathan', 'I', 'Cecilia', 'I'), 
         ('Cecilia', 'I', 'Alex', 'I'),
         
         ('Alex', 'K', 'Jonathan' ,'K'), 
         ('Jonathan', 'K', 'Cecilia', 'K'), 
         ('Cecilia', 'K', 'Alex', 'K'),

         ('Alex', 'K', 'Jonathan' ,'I'), 
         ('Jonathan', 'K', 'Cecilia', 'I'), 
         ('Cecilia', 'K', 'Alex', 'I'),

         ('Alex', 'Y', 'Jonathan' ,'OP'), 
         ('Jonathan', 'Y', 'Cecilia', 'OP'), 
         ('Cecilia', 'Y', 'Alex', 'OP'),   

         ('Alex', 'OV', 'Jonathan' ,'OV'), 
         ('Jonathan', 'OV', 'Cecilia', 'OV'), 
         ('Cecilia', 'OV', 'Alex', 'OV'),  

         ('Alex', 'r', 'Jonathan' ,'r'), 
         ('Jonathan', 'r', 'Cecilia', 'r'), 
         ('Cecilia', 'r', 'Alex', 'r'),    

         ('Alex', 'r', 'Jonathan' ,'OP'), 
         ('Jonathan', 'r', 'Cecilia', 'OP'), 
         ('Cecilia', 'r', 'Alex', 'OP'),  

         ('Alex', 'r', 'Alex' ,'OP'), 
         ('Jonathan', 'r', 'Jonathan', 'OP'), 
         ('Cecilia', 'r', 'Cecilia', 'OP'),  
         ]
for comp in COMPS:
    ref = labeled[comp[0:2]]
    scored = labeled[comp[2:4]]
    my_score = score(ref.reshape((1,1,17,3)), scored.reshape((1,1,17,3)), np.zeros((1,1,17,3)), np.zeros((1,1,17,3)))

    print(f'Reference: {comp[0]} {comp[1]}. Tested: {comp[2]} {comp[3]}. Score: {my_score}')