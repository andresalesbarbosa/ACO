
# coding: utf-8

# In[ ]:

import pants,math,random,os,re



# Function to calculate the euclidean distance between to points

def euclidean(a, b):
    return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))

# Function to calculate the exact distance between to geopoints

def geo(a,b):
    latitude = []
    longitude = []
    PI = 3.141592
    deg = int(a[0])
    _min = a[0]- deg
    rad = PI * (deg + 5.0 * _min/ 3.0) / 180.0
    latitude.append(rad)
    
    PI = 3.141592
    deg = int(b[0])
    _min = b[0]- deg
    rad = PI * (deg + 5.0 * _min/ 3.0) / 180.0
    latitude.append(rad)
    
    PI = 3.141592
    deg = int(a[1])
    _min = a[1]- deg
    rad = PI * (deg + 5.0 * _min/ 3.0) / 180.0
    longitude.append(rad)
    
    PI = 3.141592
    deg = int(b[1])
    _min = b[1]- deg
    rad = (PI * (deg + 5.0 * _min/ 3.0) / 180.0)
    longitude.append(rad)
    
    RRR = 6378.388
    q1 = math.cos( longitude[0] - longitude[1] )
    q2 = math.cos( latitude[0] - latitude[1] )
    q3 = math.cos( latitude[0] + latitude[1] )
    dij =  int( RRR * math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0)
    return dij


def read_tsp_data(tsp_name):
    tsp_name = tsp_name
    with open(tsp_name) as f:
        content = f.read().splitlines()
        cleaned = [x.lstrip() for x in content if x != ""]
        f.close()
        return cleaned


def detect_dimension(in_list):
    non_numeric = re.compile(r'[^\d]+')
    for element in in_list:
        if element.startswith("DIMENSION"):
            return non_numeric.sub("",element)


def get_cities(list,dimension,cities_set):
    dimension = int(dimension)
    for item in list:
        for num in range(1, dimension + 1):
            if item.startswith(str(num)):
                index, space, rest = item.partition(' ')
                if rest not in cities_set:
                    cities_set.append(rest)
    return cities_set


"""
Brake each coordinate 33.00 44.00 to a tuple ('33.00','44.00')
"""

def city_tup(cities_tups,list):
    for item in list:
        first_coord, second_coord = item.split(' ')
        cities_tups.append((float(first_coord), float(second_coord)))
    return cities_tups


def produce_final(file):
    cities_set = []
    cities_tups = []
    cities_dict = {}
    data = read_tsp_data(file)
    dimension = detect_dimension(data)
    cities_set = get_cities(data,dimension,cities_set)
    cities_tups = city_tup(cities_tups,cities_set)
    return cities_tups

#build test files

def main():
    burma14 = produce_final('burma14.tsp')
    #berlin52 = produce_final('berlin52.tsp')
    #rd100 = produce_final('rd100.tsp')
    #tsp225 = produce_final('tsp225.tsp')
    #pcb442 = produce_final('pcb442.tsp')
    #u574 = produce_final('u574.tsp')
    #u1060 = produce_final('u1060.tsp')
    #vm1084 = produce_final('vm1084.tsp')

    #expanded test
    #ants = [50,100,200]#,100,200,500,1000,1500]10
    #alphas = [0,7]#,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7
    #betas = [7,8,9]#0,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7
    #rhos = [0.8]#0.2,0.4,0.5,0.6,0.8
    #elites = [0.6]#0,0.2,0.4,0.5,0.6,0.8
    #limits = [100,500,1000,2000,5000,10000]#10,

    #original test
    ants = [50]#,100,200,500,1000,1500]10
    #alphas = [0,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7]
    alphas = [7,6,5,4.5,4,3.5,3,2.5,2,1.5,1,0.75,0.5,0.25,0]
    betas = [0,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7]#
    rhos = [0.2,0.4,0.5,0.6,0.8]#
    elites = [0,0.2,0.4,0.5,0.6,0.8]#
    limits = [100]#10,,500,1000,2000,5000,10000
    #clean.iloc[662030] first result from old data
    #clean.iloc[672624] last result from old data
    #wipe this data from new df

    #intest worlds
    world_burma14 = pants.World(burma14,geo)
    #world_berlin52 = pants.World(berlin52,euclidean)
    #world_rd100 = pants.World(rd100,euclidean)

	#not tested worlds
    #world_tsp225 = pants.World(tsp225,euclidean)
    #world_pcb442 = pants.World(pcb442,euclidean)
    #world_u574 = pants.World(u574,euclidean)
    #world_u1060 = pants.World(u1060,euclidean)
    #world_vm1084 = pants.World(vm1084,euclidean)
    worlds = [world_burma14]#,world_berlin52,world_rd100,world_tsp225,world_pcb442,world_u574,world_u1060,world_vm1084]
    names = ['burma14.test']#,'berlin52.test','rd100.test']#,'tsp225.test','pcb442.test','u574.test','u1060.test','vm1084.test']
    
    for limit in limits:
        for ant in ants:
            for alpha in alphas:
                for beta in betas:
                    for rho in rhos:
                        for elite in elites:
                            for index,world in enumerate(worlds):
                                med = 0
                                for i in range(30):
                                    solver = pants.Solver(ant_count=ant,alpha=alpha,beta=beta,rho=rho,elite=elite,limit=limit)
                                    solution = solver.solve(world)
                                    med += solution.distance 
                                    #print(solution.distance)
                                    fo = open(names[index],'a')
                                    string = 'ID: {i}\tant: {0}\talpha: {1}\tbeta: {2}\trho: {3}\telite: {4}\tlimit: {5}\tdistance: {6}\n'.format(ant,alpha,beta,rho,elite,limit,solution.distance,i=i)
                                    fo.write(string)
                                    fo.close()
                                fo = open(names[index],'a')
                                string = 'Med \tant: {0}\talpha: {1}\tbeta: {2}\trho: {3}\telite: {4}\tlimit: {5}\tdistance: {6}\n'.format(ant,alpha,beta,rho,elite,limit,int(med/30),i=i)
                                fo.write(string)
                                fo.close()
                                print('World: '+names[index]+' '+string)
    print('finish')
    
main()

