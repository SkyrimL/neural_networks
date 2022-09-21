import numpy
import numpy as np
import math

#读入csv文件，出来是大list套小list，小list里每个元素都是int
def read_csv_to_list(infile):
    text = []
    for i in infile:
        text.append(list(i.strip('\n').split(',')))
    for j in range(len(text)):
        for k in range(len(text[j])):
            text[j][k]=int(text[j][k])
    return(text)

#把上一个list的特征向量读出来，把放在第一个的y去掉，替换成x0=1
def get_x(infile:list):
    x=[]
    for i in range(len(infile)):
        infile[i][0]=1
        x.append(infile[i])
    return x

#把上一个list的分裂结果读出来,就是一个列表，里面的元素可以是1，2，3
def get_y(infile:list):
    y=[]
    for i in range(len(infile)):
        y.append(infile[i][0])
    return y

#把y转成onehot编码，规则自己看咋循环的吧
def onehot_y(y:list):
    onehot=[]
    for i in range(len(y)):
        if y[i]==0:
            onehot.append([1,0,0,0])
        if y[i]==1:
            onehot.append([0,1,0,0])
        if y[i]==2:
            onehot.append([0,0,1,0])
        if y[i]==3:
            onehot.append([0,0,0,1])
    return onehot

#初始化一个全是0的阿尔法的weight，attribute有几维，这个weight就有几行（还是列？没太搞清楚）
#node的数量是个超参数，我们指定中间截点到底是几个，这个weitght就有几列，可不是方阵啊
def initial_alfa_zero(listlength:int,nodes:int):
    x=[]
    for i in range(nodes):
        x.append(0)
    y=[]
    for i in range(listlength):
        y.append(x)

    return y

#算a1,a2,a3....对于没一个样本，都进行一次计算，因此就是一个list，里面有和attribute数量等同的元素
#传进去的参数 那个attribute 应该是attribute大list的一个元素，比如attribute[0]
def get_a_value(attribute:list,alpha:list):
    attribute=np.array(attribute)
    alpha=np.array(alpha)
    a=np.dot(attribute,alpha)
    return a

#从a到z的那一步，这次作业不用tanh了，用sigm
def sigm(x):
    y=1/(1+np.exp(-x))
    return y


#从a到z的那一步，这次作业不用tanh了，用sigm
#z比a还要再多加一个z0，所以列表长度又加1
def get_z_value(a:list):
    b=[1]
    for i in range(len(a)):
        b.append(sigm(a[i]))

    return b

#初始化一个全是0的贝塔的weight，因为最后分出来的y有4类，所以这块长度是4
#这次的贝塔是个矩阵了，不是个向量
def initial_beta_zero(listlength:int):
    x=[]
    for i in range(4):
        x.append(0)
    y=[]
    for i in range(listlength):
        y.append(x)

    return y

#反正y一定是四个，不写循环了。。。
def get_y_value(b:list):
    y1=np.exp(b[0])/(np.exp(b[0])+np.exp(b[1])+np.exp(b[2])+np.exp(b[3]))
    y2 = np.exp(b[1]) / (np.exp(b[0]) + np.exp(b[1]) + np.exp(b[2]) + np.exp(b[3]))
    y3 = np.exp(b[2]) / (np.exp(b[0]) + np.exp(b[1]) + np.exp(b[2]) + np.exp(b[3]))
    y4 = np.exp(b[3]) / (np.exp(b[0]) + np.exp(b[1]) + np.exp(b[2]) + np.exp(b[3]))
    y=[y1,y2,y3,y4]
    return y

#这块相当于是拿真实的y和yhat预估值去算
def Cross_entropy(yhat:list,result:list):
    return -(result[0]*math.log(yhat[0])+result[1]*math.log(yhat[1])+result[2]*math.log(yhat[2])+result[3]*math.log(yhat[3]))

#这个和下面的偏导都是，你求贝塔的偏导，肯定和他自己没关系，但是你需要知道他这个矩阵多长多宽，所以你还是得把他传进去
def gradient_beta(Beta:list,z:list,loss:list):
    vector=[]
    for j in range(len(Beta)):
        vector.append(z[j]*loss)

    return np.array(vector).T

def gradient_alpha(alpha:list,Beta:list,z:list,loss:list,attribute:list):
    gradient = []
    for i in range(len(alpha)):
        vector=[]
        for j in range(len(alpha[0])):
            a=0
            a+=loss[0]*Beta[j+1][0]+loss[1]*Beta[j+1][1]+loss[2]*Beta[j+1][2]+loss[3]*Beta[j+1][3]
            vector.append(a*z[j+1]*(1-z[j+1])*attribute[i])
        gradient.append(vector)

    return np.array(gradient).T


def Adagrad_s(s:list,gradient:list):
    gradient=np.array(gradient)
    gg=gradient*gradient
    s=np.array(s)
    news=s+gg
    return news

def Adagrad_decline(s,gradient,theta,learning_rate):
    s = np.array(s)
    s=s+1e-5
    a=np.sqrt(s)
    step=learning_rate/a
    decline=step*gradient
    new_theta=np.array(theta).T-decline

    return new_theta





infile = open(r"C:\Users\Louis\Desktop\hw5\hw5\data\tinyTrain.csv")
infile1=read_csv_to_list(infile)

nodes=4

result=get_y(infile1)

attribute=get_x(infile1)

alfa=initial_alfa_zero(len(attribute[0]),nodes)

a=get_a_value(attribute[0],alfa)

z=get_z_value(a)

Beta=initial_beta_zero(len(z))

#a和b的计算方法一样的
b=get_a_value(z,Beta)

#这个是估出来的yhat
y=get_y_value(b)

onehot_result=onehot_y(result)

entropy=Cross_entropy(y,onehot_result[0])

y=numpy.array(y)
onehot_result[0]=numpy.array(onehot_result[0])
db=y-onehot_result[0]

grad=gradient_beta(Beta,z,db)

grad_a=gradient_alpha(alfa,Beta,z,db,attribute[0])

s0=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]

sb0=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

s1=Adagrad_s(s0,grad_a)

sb1=Adagrad_s(np.array(sb0).T,grad)

new_Alpha=Adagrad_decline(s1,grad_a,alfa,0.1)

new_Beta=Adagrad_decline(sb1,grad,Beta,0.1)

a111=get_a_value(attribute[1],new_Alpha.T)

z111=get_z_value(a111)

b111=get_a_value(z,new_Beta.T)

y111=get_y_value(b111)

entropy111=Cross_entropy(y111,onehot_result[1])

y111=numpy.array(y111)
onehot_result[1]=numpy.array(onehot_result[1])
db111=y111-onehot_result[1]

grad111=gradient_beta(new_Beta.T,z111,db111)

grad_a111=gradient_alpha(new_Alpha.T,new_Beta.T,z111,db111,attribute[1])

s111=Adagrad_s(s1,grad_a111)

sb111=Adagrad_s(sb1,grad111)

new_Alpha=Adagrad_decline(s111,grad_a111,new_Alpha.T,0.1)

new_Beta=Adagrad_decline(sb111,grad111,new_Beta.T,0.1)

print(new_Alpha)
print(new_Beta)
