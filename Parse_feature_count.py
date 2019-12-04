import numpy as np
import os
from xml.etree import ElementTree as ET

# 数据路径
rootpath = r'F:\yutian\Datasets\InterceptedDatasets\randomsample'
targetpath = r'F:\yutian\Datasets\InterceptedDatasets\randomsample_encoding_feature_count'

#ONEHOT_SIZE = 38
ALL_UNIT_TYPES = ['RESOURCE', 'BASE', 'BARRACKS', 'WORKER', 'LIGHT', 'RANGED', 'HEAVY']  # 七位来编码单位类型

np.set_printoptions(threshold=np.inf)


# 编码函数,对每个传入的文件进行编码，返回三维数组map和winner
def encoding(r_path,file):
    map = np.ndarray((8, 8, 7))
    map.fill(0)
    # 数组存放统计次数
    unittype_count = [0 for i in range(7)]
    unitowner_count = [0 for i in range(2)]
    resources_count = [0 for i in range(20)]
    hitpoints_count = [0 for i in range(10)]
    actiontype_count = [0 for i in range(62)]

    # 解析
    tree = ET.parse(os.path.join(r_path, file))
    # 获取根节点
    root = tree.getroot()

    traceentry_node = root.find('rts.TraceEntry')
    pgs_node = traceentry_node.find('rts.PhysicalGameState')
    # 获取winner
    winner_node = root.find('winner')
    winner = int(winner_node.get('winner'))
    # print("winner:")
    # print(winner)
    # 平局情况
    if winner == -1:
        # 第一种，平局为winner0
        # winner=0
        # 第二种，平局为winner1
        # winner=1
        # 第三种，随机
        # '''
        r = np.random.randint(0, 2)
        if r == 0:
            winner = 0
        if r == 1:
            winner = 1
        # '''
    # print(winner)

    # 遍历第一遍统计
    # <units>下的数据
    units_node = pgs_node.find('units')
    for unit_node in units_node:
        # 统计unittype的频次
        type = unit_node.get('type')
        i = ALL_UNIT_TYPES.index(type.upper())
        unittype_count[i] += 1

        # 统计unitowner的频次
        player = unit_node.get('player')
        if player == '0':
            unitowner_count[0] += 1
        elif player == '1':
            unitowner_count[1] += 1

        # 统计resources的频次
        resources = int(unit_node.get('resources'))
        if resources == 0:  # 不用统计
            pass
        else:
            resources_count[resources - 1] += 1

        # 统计hitpoints的频次
        hitpoints = int(unit_node.get('hitpoints'))
        if hitpoints == 0:  # 不用统计
            pass
        else:
            hitpoints_count[hitpoints - 1] += 1

        # <actions>下的数据
    actions_node = traceentry_node.find('actions')
    for action_node in actions_node:
        # 统计actiontype的频次
        a_type = int(action_node.find('UnitAction').get('type'))
        actiontype_count[a_type] += 1

    # 输出统计结果
    '''
    print(unittype_count)
    print(unitowner_count)
    print(resources_count)
    print(hitpoints_count)
    print(actiontype_count)
    '''
    # 开始根据频次生成三维数组map
    for unit_node in units_node:
        x = int(unit_node.get('x'))
        y = int(unit_node.get('y'))
        # unittype位
        type = unit_node.get('type')
        i = ALL_UNIT_TYPES.index(type.upper())
        #print(unittype_count[i])
        map[y, x, 0] = unittype_count[i]

        # unitowner位
        player = unit_node.get('player')
        if player == '0':
            map[y, x, 1] = unitowner_count[0]
        elif player == '1':
            map[y, x, 1] = unitowner_count[1]

        # resources位
        resources = int(unit_node.get('resources'))
        if resources == 0:
            map[y, x, 2] = 0
        else:
            map[y, x, 2] = resources_count[resources - 1]

        # hitpoints位
        hitpoints = int(unit_node.get('hitpoints'))
        if hitpoints == 0:
            map[y, x, 3] = 0
        else:
            map[y, x, 3] = hitpoints_count[hitpoints - 1]

    if actions_node:
        for action_node in actions_node:
            unitid = action_node.get('unitID')
            # 查找有action的unit
            for unit_node in units_node:
                id = unit_node.get('ID')
                if unitid == id:
                    a_x = int(unit_node.get('x'))
                    a_y = int(unit_node.get('y'))

            # actiontype位
            a_type = int(action_node.find('UnitAction').get('type'))
            map[a_y, a_x, 4] = actiontype_count[a_type]

            b_x = a_x
            b_y = a_y
            if a_type == 5:
                # x,y
                b_x = int(action_node.find('UnitAction').get('x'))
                b_y = int(action_node.find('UnitAction').get('y'))
            else:
                # parameter
                a_parameter = int(action_node.find('UnitAction').get('parameter'))
                if a_parameter == 0:  # 上
                    b_y = b_y - 1
                elif a_parameter == 1:  # 右
                    b_x = b_x + 1
                elif a_parameter == 2:  # 下
                    b_y = b_y + 1
                elif a_parameter == 3:  # 左
                    b_x = b_x - 1

            # 被动承受位
            if a_type > 0:
                map[b_y, b_x, 5] = actiontype_count[a_type]
            # 生成unittype位
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