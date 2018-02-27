
import pydot
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn.utils

encoder_list=[] # Store all the encoded data from the inputs
predicted_list=[] # List to store all the predicted values predicted by the tree
edge_dict={} # Dictionary to store the all edges of the tree
result=["No","Yes"]
no_of_rowsTrain=2000 # Number of Training data
no_of_rowsTest=500 # Number of Testing data
file="chess.csv" # Name of the file that is used for the making the tree
graph_output="graph_output.jpg" # Name of the output of the graph which contains tree


def partition(a):
    """Partitions according to the unique elements"""
    return {c:(a==c).nonzero()[0]for c in np.unique(a)}


def data_processing(inputarray):
    """Encodes the data coming from the file
    Encodes in to 0,1,2 according to the number of attributes
    """
    labelEncoder_X=LabelEncoder()
    for i in range(np.size(inputarray,1)):
        input=np.ravel(inputarray[:,i])
        y=labelEncoder_X.fit_transform(input)
        encoder_list.append(y)
    np.asmatrix(encoder_list)    
    return np.transpose(encoder_list)



def element_count(columns):
    """Calculates the probability
    """
    count_list=[]
   
    for i in range(np.amax(columns)+1):
        count=0
        for j in range(len(columns)):
            if(columns[j]==i):
                 count=count+1
        count_list.append(count)
    return (np.divide(count_list,len(columns)))

def true_count(xarray,yarray):
    """
    """
    true_countlist=[]
    false_countlist=[]
    for i in range(np.amax(xarray)+1):
        true_value=0
        false_value=0
        for j in range(len(xarray)):
            if(xarray[j]==i):
                if(yarray[j]==1):
                    true_value=true_value+1
                else:
                    false_value=false_value+1
        if(true_value!=0 or false_value!=0):
            true_countlist.append(true_value)
            false_countlist.append(false_value)
    
    return(np.divide(true_countlist,np.add(true_countlist,false_countlist)))    


def entropy(pb):
    """Function which has equation of entropy
    """
    entropy_val=-(pb*np.log2(pb)+(1-pb)*np.log2(1-pb))
    return (np.round(entropy_val,4))


    
def entropy_cal(probability):
    entropy_list=[]
    for i in range(len(probability)):
        if(probability[i]==0 or probability[i]==1):
            entropy_list.append(0)
        else:    
            entropy_list.append(entropy(probability[i]))
    return entropy_list



def gain_percent(system_entropy,entropy,probability):
    """Calculates the gain percent
    """
    total=0
    for i in range(len(entropy)):
        total=total+probability[i]*entropy[i]
    return system_entropy-total

def create_tree(test,y):
            """Recursive function which keeps on appending attribute
            on dictionary with the maximum gain            
            """
            if(len(set(y))==1 or len(y) == 0 ): 
                    return ((y[0].item()))                                
            gain=[]
            prob_res=element_count(y)
            system_entropy=entropy(prob_res[1])

            """For the calculation of the Probability,Entropy"""
            for i in range(np.size(test,1)):
                if (len(set(test[:,i]))==1):
                    gain.append(0)             
                else:
                    prob_count=element_count(test[:,i])
                    prob_true=true_count(test[:,i],y)
                    entropy_list=entropy_cal(prob_true)                  
                    gain.append((gain_percent(system_entropy,entropy_list,prob_count)))                                                
            gain=np.asarray(gain)
            if np.all(gain<1e-6):
                return y
           
            max_gainindex=np.argmax(gain)
            sets = partition(test[:, max_gainindex])
            res = {}
        
            for k,v in sets.items():              
                y_subset=y.take(v,axis=0)
                x_subset=test.take(v,axis=0)
                res["%d = %d" % (max_gainindex, k)] = create_tree(x_subset, y_subset) 
                 
            return res    

def test(res,data):
    """Function which appends the predicted elements by 
    the tree to the predicted_list. 
    """
    if(isinstance(res,dict)):        
        a=res.keys()
        for i in a:
            b=i
            break
        first_element=int(b.split("=")[0])
        c=data[first_element]
        d=str(first_element)+" = "+str(c)
        for k,v in res.items():
            if(k==d):
                test(v,data)
    else: 
        predicted_list.append(res)
                               
def accuracy(res,test_data,actual_output):
    """Function to calculate TP,TN,FP,FN 
    ,F1 score and the accuracy of the tree     
    """
    #Predicted True, Actual_True
    true_positive=0    
    #Predicted True,Actual_False
    false_positive=0    
    #Predicted False,Actual False
    true_negative=0    
    #Predicted False,Actual True
    false_negative=0    
    predicted_yes=0
    predicted_no=0
    actual_no=0
    actual_yes=0
    f1=0    
    for i in range(len(test_data)):
        test(res,test_data[i])
#    print("Predicted data{}".format(list1))
    for j in range(len(predicted_list)):
        if(predicted_list[j]==1 and actual_output[j]==1):
            true_positive=true_positive+1
        if(predicted_list[j]==1 and actual_output[j]==0):
            false_positive=false_positive+1
        if(predicted_list[j]==0 and actual_output[j]==0):
            true_negative=true_negative+1
        if(predicted_list[j]==0 and actual_output[j]==1):
            false_negative=false_negative+1   
    predicted_yes=false_positive+true_positive
    predicted_no=true_negative+false_negative
    actual_yes=false_negative+true_positive
    actual_no=true_negative+false_positive
    if not predicted_yes==0 and  not actual_yes==0:
        precision=true_positive/predicted_yes
        recall=true_positive/actual_yes 
        f1=2*(precision*recall)/(precision+recall)       
    accuracy=((true_positive+true_negative)/len(predicted_list))*100
    print("True posiitive is {}".format(true_positive))
    print("False posiitive is {}".format(false_positive))
    print("True negative is {}".format(true_negative))
    print("False negative is {}".format(false_negative))   
    return "F1 score= {}".format(f1),"Accuracy= {}".format(accuracy),

def edge_finder():
    """This function appends the key value pair to the 
    dictionary which is helpful for making the edges of 
    the tree.
    """
    for i in range(len(column_headers)):        
        for j in range(len(sorted(set(df_edge.iloc[:,i])))):            
            edge_dict["%d = %d" %(i,j)]=sorted(set(df_edge.iloc[:,i]))[j]     
        
def element_return(v):
    """This function returns the node of the tree according to 
    the value of the key. Value must be type of dictionary.It 
    is useful for making the tree only.
    """
    res=v.keys()
    for i in res:
        b=i
        break
    first_node=int(b.split("=")[0])
    return first_node


def draw(parent_name, child_name,edge_name):
    """Function which draws parent node , child node
    and edges"""
    edge = pydot.Edge(parent_name, child_name)
    edge.set_label(edge_name)
    graph.add_edge(edge)


def visit(node,next_node=None):
    """This function is for drawing the tree"""
    res=node.keys()
    for i in res:
        b=i
        break
    for j in node.items():        
        first_node=int(b.split("=")[0])

    for k,v in node.items():
        if isinstance(v,dict):
            # draw_node()
            first_element=column_headers[first_node]
            second_element=column_headers[element_return(v)]
            edge_value=edge_dict[k]
            draw(first_element,second_element,edge_value)

            visit(v)
        else:            
            first_element=column_headers[first_node]
            edge_value=edge_dict[k]
            draw(first_element,v,edge_value)



"""Data Processing for making graph, splitting data to training and testing"""   
input=pd.read_csv(file)
inputarray=np.asmatrix(input)
test1 = data_processing(inputarray)
df=pd.DataFrame(test1)
df_edge=pd.DataFrame(input)
column_headers=list(df_edge)

"""Shuffling of the inputs"""
df1=sklearn.utils.shuffle(df)
x=np.array(df1.iloc[:,:-1])
y=np.array(df1.iloc[:,-1])

"""Splitting data into training and testing"""
train_dataX=x[:no_of_rowsTrain]
train_dataY=y[:no_of_rowsTrain]
test_dataX=x[-no_of_rowsTest:]
test_dataY=y[-no_of_rowsTest:]

"""Creates a decision tree according to the training data"""
c=create_tree(train_dataX,train_dataY)

"""Returns and prints the associated Accuracy"""
print(accuracy(c,test_dataX,test_dataY))

"""Below functions create the graph"""
edge_finder()#Creates a dictionary according to the edges"
graph = pydot.Dot(graph_type='digraph')
visit(c)
graph.write_png(graph_output)












        






