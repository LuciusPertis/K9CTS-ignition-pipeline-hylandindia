#from skimage.morphology import skeletonize
#from skimage import draw

from typing import Tuple
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import math as m
img_id=4

class Graph:
    NodeMap_intoc=dict()
    NodeMap_ctoin=dict()
    parent=dict()
    EdgeList=list()
    def addParent(self,i1:int,i2:int):
        self.parent[i2]=self.parent[i1]
    def addNode(self,i:int,c:Tuple):
        #print(i,c)
        self.NodeMap_intoc[i]=c
        self.NodeMap_ctoin[c]=i
        self.parent[i]=0
    def addEdge(self,c1:Tuple,c2:Tuple,w:float):
        v1,v2=self.NodeMap_ctoin[c1],self.NodeMap_ctoin[c2]
        if isinstance(v1,tuple):
            print("tuple error",v1)
        self.EdgeList.append((v1,v2,w))
        self.EdgeList.append((v2,v1,w))
        #self.addEdge(v1,v2,w)
        self.addParent(int(v1),v2)
        '''if isinstance(v2,tuple):
            print("*",c1,c2,w)
        self.parent[v2]=v1'''
    '''def addEdge(self,i1:int,i2:int,w:float):
        self.EdgeList.append((i1,i2,w))
        self.EdgeList.append((i2,i1,w))
        self.addParent(int(i1),i2)'''
    def assignGraph(self,g):
        self.Graph=g
    def createAdjacent(self):
        n=len(self.NodeMap_intoc)
        adj=np.zeros((n,n))
        for i in self.EdgeList:
            x,y,w=i
            adj[x,y]=w
            adj[y,x]=w
        return adj
    def __init__(self) -> None:
        pass

def display_matrix_img(mat):
    plt.imshow(mat,cmap='binary',interpolation='nearest')
    #plt.imshow(mat,interpolation='nearest')
    #plt.matshow(mat,cmap='binary')
    plt.show()
def circ_search(x,y,X,Y):
    x1,x2=X
    y1,y2=Y
    if x<x2-1 and y==y1:
        x+=1
    elif x==x2-1 and y<y2-1:
        y+=1
    elif x<=x2-1 and y==y2-1 and x>x1:
        x-=1
    elif x==x1 and y<=y2-1 and y>0:
        y-=1
    else:
        return False,x,y
    return True,x,y
def srch_node(G,x,y,radius=1):
    v=np.zeros((2*radius+1,2*radius+1))
    g=G[x-radius:x+radius+1,y-radius:y+radius+1]
    if len(g)==0 or len(g)>=len(G)//5 or g.size==0:
        return False,x,y
    #print(g)
    i,j=0,0
    #if True:
    try:
        while g[i,j]==0 and v[i,j]==0:
            v[i,j]=1
            k,i,j=circ_search(i,j,(0,2*radius+1),(0,2*radius+1))
            if not k:
                break
            if g[i,j]>=1:
                return True,x+i-radius,y+j-radius
        else:
            if g[i,j]>=1:
                return True,x+i-radius,y+j-radius
    except:
        print(":",i,j,g)
    return srch_node(G,x,y,radius+1)

'''def srch_nodeP(G,x,y,radius=1):
    v=np.zeros((2*radius+1,2*radius+1))
    g=G.Graph[x-radius:x+radius+1,y-radius:y+radius+1]
    if len(g)==0 or len(g)>=len(G.Graph)//5:
        return False,x,y
    print("r",radius,x,y)
    i,j=0,0
    if True:
    #try:
        while not ((x+i-radius,y+j-radius) in G.NodeMap_ctoin ):
            v[i,j]=1
            k,i,j=circ_search(i,j,(0,2*radius+1),(0,2*radius+1))
            if not k:
                break
        z=G.NodeMap_ctoin[(x+i-radius,y+j-radius)]
        while g[i,j]==1 and G.parent[z]!=-1 and v[i,j]==0:
            v[i,j]=1
            k,i,j=circ_search(i,j,(0,2*radius+1),(0,2*radius+1))
            while not ((x+i-radius,y+j-radius) in G.NodeMap_ctoin ):
                v[i,j]=1
                k,i,j=circ_search(i,j,(0,2*radius+1),(0,2*radius+1))
                if not k:
                    break
            z=G.NodeMap_ctoin[(x+i-radius,y+j-radius)]
            if not k:
                break
            if g[i,j]>=1 and G.parent[z]==-1:
                return True,x+i-radius,y+j-radius
        else:
            if g[i,j]>=1 and G.parent[z]==-1:
                return True,x+i-radius,y+j-radius    
    #except:
        #print("=",i,j)
    return srch_nodeP(G,x,y,radius+1)'''
def srch_nodeP(list,g,x,y):
    for i in list:
        if i[0] in range(max(x-7,0),x+8) and i[1] in range(max(y-7,0),y+8):
            z=g.NodeMap_ctoin[i]
            #print("r",i)
            if g.parent[z]==-1:
                return True,i[0],i[1]
    return False,x,y

os.chdir(r'C:\Users\Anshman Dhan\Pictures')
img_fname=os.path.join('map samples',f'map_sam{img_id}.jpg') 
image=imread(img_fname)

# Change RGB color to gray 
image=rgb2gray(image)

# Change gray image to binary
image=np.where(image>np.mean(image),1.0,0.0)
#print(image)
# perform skeletonization
display_matrix_img(image)
image=image.astype(int)
print(np.unique(image,return_counts=True))

#skeleton = skeletonize(image)
#display_matrix_img(skeleton)
print("hi")
visit=np.zeros(image.shape)
X,Y=image.shape
x,y=0,0
plt.gca().invert_yaxis()

while image[x,y]==1 and visit[x,y]==0:
    visit[x,y]=1
    print(x,y)
    '''if x<X-1 and y==0:
        x+=1
    elif x==X-1 and y<Y-1:
        y+=1
    elif x<=X-1 and y==Y-1 and x>0:
        x-=1
    elif x==0 and y<=Y-1 and y>0:
        y-=1'''
    k,x,y=circ_search(x,y,(0,X),(0,Y))
    if  not k:
        break

print(image)
print(x,y,image[x-1:x+1,y])
srch_node(image,x,y)
visit=np.zeros(image.shape)
graph=np.zeros(image.shape)
g=Graph()
g.addNode(1,(x,y))
#cv2.imshow('t',image)
graph[x,y]=1
visit[x,y]=1
x_1,y_1=x,y
if x==0:
    x=1
elif x==X-1:
    x==X-2
if y==0:
    y=1
elif y==Y-1:
    y=Y-2
#Temp=np.zeros((32,3,3))
#print(Temp)

stack=list()
graph[x,y]=1
g.addNode(2,(x,y))
g.parent[1]=-1
g.addEdge((x_1,y_1),(x,y),m.sqrt((x_1-x)**2+(y_1-y)**2))
i=5
stack.append((x,y))
j=3
X,Y=image.shape

while len(stack)>0:
    s=stack.pop()
    i-=1
    x,y=s
    visit[x,y]=1
    plt.scatter(y,x,c=[1],alpha=0.5,s=2)
    print(x,y)
    x1,x2=x-1,x+1
    y1,y2=y-1,y+1
    if x1>=0 and image[x1,y]==0 and visit[x1,y]==0:
        stack.append((x1,y))
    if x2<X and image[x2,y]==0 and visit[x2,y]==0:
        stack.append((x2,y))
    if y1>=0 and image[x,y1]==0 and visit[x,y1]==0:
        stack.append((x,y1))
    if y2<Y and image[x,y2]==0 and visit[x,y2]==0:
        stack.append((x,y2))
    if i==0:
        i=3
        graph[x,y]=1
        g.addNode(j,(x,y))
        j+=1
        coord_x=[x]
        coord_y=[y]
        #try:
        if True:
            k,x_1,y_1=srch_node(graph,x,y)
            if k:
                coord_x.append(x_1)
                coord_y.append(y_1)
                g.addEdge((x_1,y_1),(x,y),m.sqrt((x_1-x)**2+(y_1-y)**2))
                #plt.plot(coord_y,coord_x,'-y')
        #except:
        else:
            print("+",x,y)
plt.show()
#print(g.parent)
#print(sorted(g.NodeMap_ctoin.values()))
#x,y=50,50
#srch_node(graph,x,y)  
#display_matrix_img(graph)
x,y=g.NodeMap_intoc[1]
nodelist=sorted(g.NodeMap_ctoin,key=lambda a:m.sqrt((a[0]-x)**2+(a[1]-y)**2) )
'''a=g.NodeMap_ctoin[(1,47)]
for i in g.EdgeList:
    if a in i[:2]:
        print(i)'''
g.assignGraph(graph)
for i in nodelist:
    #print("i",i)
    index=g.NodeMap_ctoin[i]
    plt.scatter(i[1],i[0],c=[1],alpha=0.5,s=2)
    if g.parent[index]==-1:
        continue
    k,x,y=srch_nodeP(nodelist,g,i[0],i[1])
    if k:
        #print("P",x,y)
        g.addEdge((x,y),i,m.sqrt((i[0]-x)**2+(i[1]-y)**2))
#print(g.parent)
ax=plt.axes()
plt.gca().invert_yaxis()
for i in g.EdgeList:
    coord_y=[g.NodeMap_intoc[i[j]][1] for j in [0,1] ]
    coord_x=[g.NodeMap_intoc[i[j]][0] for j in [0,1] ]
    ax.scatter(coord_y,coord_x,c=[1,1],alpha=0.5)
    ax.plot(coord_y,coord_x,'-y')
plt.show()