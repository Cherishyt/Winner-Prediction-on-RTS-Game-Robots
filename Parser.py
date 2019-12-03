from xml.etree import ElementTree as ET
import numpy as np
import os


#数据路径
rootpath=r'F:\yutian\Datasets\InterceptedDatasets\sample_part_20\samplepart1'
targetpath=r'F:\yutian\Datasets\InterceptedDatasets\sample_part_20_encoding\samplepart1'
ONEHOT_SIZE=38
ALL_UNIT_TYPES = ['RESOURCE','BASE', 'BARRACKS', 'WORKER', 'LIGHT', 'RANGED','HEAVY']#七位来编码单位类型
ALL_RESOURCES=[1,2,3,4,5]
ALL_HP=[1,2,3,4]
ALL_ACTION_TYPES=[0,1,2,3,4,5]
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    def encoding(width,height,units_node,actions_node,p_resources):
        map = np.ndarray((int(width), int(height),ONEHOT_SIZE))#创建三维数组（w,h,38）
        map.fill(0)
        #编码units数据
        unit_len=len(units_node)
        for unit_node in units_node:
            x=int(unit_node.get('x'))
            y=int(unit_node.get('y'))
            #type,0-6
            type=unit_node.get('type')
            i = ALL_UNIT_TYPES.index(type.upper())
            map[y,x,i] = 1

            #player,7-8
            player=unit_node.get('player')
            if player=='0':
                map[y,x,7]=1
                if type=='Base':
                    unit_node.set('resources',p_resources[0])
                #a=ALL_RESOURCES.index(int(p_resources[0]))

            elif player=='1':
                map[y,x,8]=1
                if type=='Base':
                    unit_node.set('resources',p_resources[1])
                #a=ALL_RESOURCES.index(int(p_resources[1]))

            #resources,9-15
            #单独编码Base的resources值
            #if type=='Base':
            #    map[y,x,9+a]=1
            #else:
            resources=int(unit_node.get('resources'))
            if resources==0:#不用编码
                pass
            elif resources<6:
                j=ALL_RESOURCES.index(resources)
                map[y,x,9+j]=1#编码位为9,10,11,12,13
            elif resources<10:
                map[y,x,14]=1
            elif resources>=10:
                map[y,x,15]=1

            #hitpoints,16-20
            hitpoints=int(unit_node.get('hitpoints'))
            if hitpoints==0:#不用编码
                pass
            elif hitpoints<5:
                k=ALL_HP.index(hitpoints)
                map[y,x,16+k]=1#编码位为16,17,18,19
            elif hitpoints>=5:
                map[y,x,20]=1

        #编码action数据
        if actions_node:
            for action_node in actions_node:
                unitid=action_node.get('unitID')

                #查找有action的unit
                for unit_node in units_node:
                    id=unit_node.get('ID')
                    if unitid==id:
                        a_x=int(unit_node.get('x'))
                        a_y=int(unit_node.get('y'))

                #type,21-26
                a_type=int(action_node.find('UnitAction').get('type'))
                l=ALL_ACTION_TYPES.index(a_type)
                map[a_y,a_x,21+l]=1

                b_x=a_x
                b_y=a_y
                if a_type==5:
                    #x,y
                    b_x=int(action_node.find('UnitAction').get('x'))
                    b_y=int(action_node.find('UnitAction').get('y'))
                else:
                    #parameter
                    a_parameter=int(action_node.find('UnitAction').get('parameter'))
                    if a_parameter==0:#上
                        b_y=b_y-1
                    elif a_parameter==1:#右
                        b_x=b_x+1
                    elif a_parameter==2:#下
                        b_y=b_y+1
                    elif a_parameter==3:#左
                        b_x=b_x-1
                l=l-1
                if l>-1:#被动承受位，27-31
                    map[b_y,b_x,27+l]=1
                if a_type==4:#生成单元类型位,32-37
                    #unitType
                    b_type=action_node.find('UnitAction').get('unitType')
                    m=ALL_UNIT_TYPES.index(b_type.upper())-1
                    map[b_y,b_x,32+m]=1
        #print(map)
        return map

    if not os.path.exists(targetpath):
        os.mkdir(targetpath)

    npy_path=targetpath+'./npy'
    #txt_path=targetpath+'./txt'
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    '''
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    '''
    for file in os.listdir(rootpath):#遍历所有xml文件
        #获取文件名称
        filename=os.path.splitext(file)[0]
        print(filename)

        #解析
        tree=ET.parse(os.path.join(rootpath, file))
        #获取根节点
        root=tree.getroot()

        #获取winner
        winner_node=root.find('winner')
        winner=int(winner_node.get('winner'))
        #print("winner:")
        #print(winner)
        #平局情况
        if winner==-1:
            #第一种，平局为winner0
            #winner=0
            #第二种，平局为winner1
            #winner=1
            #第三种，随机
            #'''
            r=np.random.randint(0,2)
            if r==0:
                winner=0
            if r==1:
                winner=1
            #'''
        #print(winner)

        #获取待编码的数据
        traceentry_node=root.find('rts.TraceEntry')
        #time
        time=traceentry_node.get('time')
        #print("time="+time)
        pgs_node=traceentry_node.find('rts.PhysicalGameState')
        #width和height
        width=pgs_node.get('width')
        height=pgs_node.get('height')
        #player0和player1的resources值
        p_resources=[0,0]
        players_node=pgs_node.find('players')
        i=0
        for player_node in players_node:
            p_resources[i]=player_node.get('resources')
            #print(p_resources[i])
            i+=1
        #<units>下的数据
        units_node=pgs_node.find('units')
        #<actions>下的数据
        actions_node=traceentry_node.find('actions')
        #编码
        x=encoding(width,height,units_node,actions_node,p_resources)
        #print(x)

        #写入npy文件

        newfilename=filename+'-'+str(winner)+'.npy'
        newfilepath=os.path.join(npy_path, newfilename)
        print(newfilepath)
        np.save(newfilepath,x)

        #写入TXT文件
        '''
        newtxtname=filename+'-'+str(winner)+'.txt'
        newtxtpath=os.path.join(txt_path,newtxtname)
        print(newtxtpath)
        test=np.load(newfilepath,encoding="latin1")
        newtxt=open(newtxtpath,'w')
        print(test,file=newtxt)
        '''












