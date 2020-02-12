import numpy as np
import os
from xml.etree import ElementTree as ET

#the path of reading dataset
rootpath = r'F:\yutian\Datasets\InterceptedDatasets\randomsample'
#the path of saving encoding npy files
targetpath = r'F:\yutian\Datasets\InterceptedDatasets\randomsample_encoding_feature_count'

ALL_UNIT_TYPES = ['RESOURCE', 'BASE', 'BARRACKS', 'WORKER', 'LIGHT', 'RANGED', 'HEAVY'] 

np.set_printoptions(threshold=np.inf)


# encoding function
def encoding(r_path,file):
    map = np.ndarray((8, 8, 7))
    map.fill(0)
	
    unittype_count = [0 for i in range(7)]
    unitowner_count = [0 for i in range(2)]
    resources_count = [0 for i in range(20)]
    hitpoints_count = [0 for i in range(10)]
    actiontype_count = [0 for i in range(6)]

    tree = ET.parse(os.path.join(r_path, file))
    root = tree.getroot()

    traceentry_node = root.find('rts.TraceEntry')
    pgs_node = traceentry_node.find('rts.PhysicalGameState')
    # get labels
    winner_node = root.find('winner')
    winner = int(winner_node.get('winner'))
    # print("winner:")
    # print(winner)
    # draw
    if winner == -1:
        # winner=0
        # winner=1
        # random
        # '''
        r = np.random.randint(0, 2)
        if r == 0:
            winner = 0
        if r == 1:
            winner = 1
        # '''
    # print(winner)

    # <units>
    units_node = pgs_node.find('units')
    for unit_node in units_node:
        # count unittype frequency
        type = unit_node.get('type')
        i = ALL_UNIT_TYPES.index(type.upper())
        unittype_count[i] += 1

        #count unitowner frequency
        player = unit_node.get('player')
        if player == '0':
            unitowner_count[0] += 1
        elif player == '1':
            unitowner_count[1] += 1

        # count resources frequency
        resources = int(unit_node.get('resources'))
        if resources == 0: 
            pass
        else:
            resources_count[resources - 1] += 1

        # count hitpoints frequency
        hitpoints = int(unit_node.get('hitpoints'))
        if hitpoints == 0: 
            pass
        else:
            hitpoints_count[hitpoints - 1] += 1

        # <actions>
    actions_node = traceentry_node.find('actions')
    for action_node in actions_node:
        # count actiontype frequency
        a_type = int(action_node.find('UnitAction').get('type'))
        actiontype_count[a_type] += 1

    #statistical results
    '''
    print(unittype_count)
    print(unitowner_count)
    print(resources_count)
    print(hitpoints_count)
    print(actiontype_count)
    '''
    # generate 3D array based on statistical results
    for unit_node in units_node:
        x = int(unit_node.get('x'))
        y = int(unit_node.get('y'))
        # unittype
        type = unit_node.get('type')
        i = ALL_UNIT_TYPES.index(type.upper())
        #print(unittype_count[i])
        map[y, x, 0] = unittype_count[i]

        # unitowner
        player = unit_node.get('player')
        if player == '0':
            map[y, x, 1] = unitowner_count[0]
        elif player == '1':
            map[y, x, 1] = unitowner_count[1]
			
        # resources
        resources = int(unit_node.get('resources'))
        if resources == 0:
            map[y, x, 2] = 0
        else:
            map[y, x, 2] = resources_count[resources - 1]

        # hitpoints
        hitpoints = int(unit_node.get('hitpoints'))
        if hitpoints == 0:
            map[y, x, 3] = 0
        else:
            map[y, x, 3] = hitpoints_count[hitpoints - 1]

    if actions_node:
        for action_node in actions_node:
            unitid = action_node.get('unitID')
			#look for action's unit by ID
            for unit_node in units_node:
                id = unit_node.get('ID')
                if unitid == id:
                    a_x = int(unit_node.get('x'))
                    a_y = int(unit_node.get('y'))

            # actiontype
            a_type = int(action_node.find('UnitAction').get('type'))
            map[a_y, a_x, 4] = actiontype_count[a_type]

            b_x = a_x
            b_y = a_y
            if a_type==0:
                b_x = a_x
                b_y = a_y
            elif a_type == 5:
                # x,y
                b_x = int(action_node.find('UnitAction').get('x'))
                b_y = int(action_node.find('UnitAction').get('y'))
            else:
                # parameter
                a_parameter = int(action_node.find('UnitAction').get('parameter'))
                if a_parameter == 0:  # up
                    b_y = b_y - 1
                elif a_parameter == 1:  # right
                    b_x = b_x + 1
                elif a_parameter == 2:  # down
                    b_y = b_y + 1
                elif a_parameter == 3:  # left
                    b_x = b_x - 1

            # passive action type
            if a_type > -1:
                map[b_y, b_x, 5] = actiontype_count[a_type]
            # produce unitType
            if a_type == 4:
                b_type = action_node.find('UnitAction').get('unitType')
                m = ALL_UNIT_TYPES.index(b_type.upper())
                map[b_y, b_x, 6] = unittype_count[m]

    print(map)
    return map,winner


if __name__ == '__main__':

    if not os.path.exists(targetpath):
        os.mkdir(targetpath)
    #Create npy folder
    npy_path = targetpath + './npy'
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    '''
    txt_path=targetpath+'./txt'
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    '''


    #read datasets
    map = np.ndarray((8, 8, 7))

    for file in os.listdir(rootpath):
        # get filename
        filename = os.path.splitext(file)[0]
        #get encoding results
        map,winner=encoding(rootpath,file)

        #write into npy_path
        newfilename = filename + '-' + str(winner) + '.npy'
        newfilepath = os.path.join(npy_path, newfilename)
        print(newfilepath)
        np.save(newfilepath,map)

        #write into txt_path
        '''
        newtxtname=filename+'-'+str(winner)+'.txt'
        newtxtpath=os.path.join(txt_path,newtxtname)
        print(newtxtpath)
        test=np.load(newfilepath,encoding="latin1")
        newtxt=open(newtxtpath,'w')
        print(test,file=newtxt)
        '''
