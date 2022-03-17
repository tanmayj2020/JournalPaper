
from webbrowser import get
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch

def get_children(model):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children



def open_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch



def get_layer_parameter(layer):
    total_layer_parameters = 0
    parameter_list = list(parameter for parameter in layer.parameters())
    for parameter in parameter_list:
        total_layer_parameters += parameter.numel() * parameter.element_size()
    return total_layer_parameters



def get_total_latency(total_size_edge , total_size_server , activation_size , c_edge , s_edge ,bandwidth , c_server , s_server):
    edge_latency = (total_size_edge * 8) / (c_edge * s_edge * (10**9))
    upload_latency = activation_size * 8 / (bandwidth * (10**6))
    server_latency = (total_size_server*8) / (c_server * s_server * (10**9))
    return edge_latency + upload_latency + server_latency


def get_activation_size(input_batch):    
    return input_batch.numel() * input_batch.element_size()


def get_activation_size_residual(input_batch ,previous_output ):    
    return (input_batch.numel() * input_batch.element_size()) + (previous_output.numel() * previous_output.element_size())



def get_pi_latency(total_size_edge , c_edge , s_edge):
    edge_latency = (total_size_edge * 8) / (c_edge * s_edge * (10**9))
    
    return edge_latency 


def get_upload_latency( activation_size , bandwidth ):
    upload_latency = (activation_size * 8 )/ (bandwidth * (10**6))
    return upload_latency 




# Resnet18 which has total 45 layers
child_dict = {
    0 : 0 , 1 : 1 , 2 : 2 , 3 : 3 , 4 : 4 , 5 : 4 , 6: 4 , 7 : 4 , 8: 4 , 9 : 4 , 10 : 4 , 11 : 4 , 12 : 4 , 13 : 4 , 14 : 5 , 15 : 5 , 16 : 5 , 17 : 5 , 18 : 5 , 19 : 5 , 20 : 5 , 21 : 5 , 22 : 5 , 23 : 5 , 24 : 6 , 25 : 6 , 26 : 6 , 27 : 6 , 28 : 6 , 29 : 6 , 30 : 6 , 31 : 6 , 32 : 6 , 33 : 6 , 34 : 7 , 35 : 7 , 36 : 7 , 37 : 7 , 38 : 7 , 39 : 7 , 40 : 7 , 41 : 7 , 42 : 7 , 43 : 7 , 44 : 8 , 45 : 9
}
block_dict = {4 : [1 , 4 , "n" ]  , 5 :[1 , 4 , "n"] , 6 : [1 , 4 , "n"] , 7 : [1 , 4 , "n"] , 8 : [1  , 4 , "n"] , 9 : [2 , 9 , "n"] , 10 : [2 , 9 , "n"] , 11 : [2 , 9 , "n"] , 12 : [2 , 9 , "n"] , 13 : [2 ,9 , "n"]  ,14 : [1 , 14 , "y"] , 15 : [1 , 14 , "y"] , 16 : [1 , 14 , "y"] , 17 : [1 , 14 , "y" ], 18 : [1 , 14 , "y"] , 19 : [2 , 19 , "n"], 20 : [2 , 19 , "n"]  , 21 : [2 , 19 , "n"] , 22 : [2 , 19 , "n"] ,  23 : [2 ,19 , "n"] , 24 : [1 , 24 , "y"] , 25 : [1 , 24 , "y"] , 26 : [1 , 24 , "y"] , 27 : [1 , 24 , "y"],  28 : [1 , 24 , "y"] , 29 : [2 , 29 ,"n"] , 30 : [2 , 29 , "n"], 31 : [2 , 29 , "n"], 32 : [2 , 29 , "n"] , 33 : [2 , 29 , "n"] , 34 : [1 , 34 , "y"] , 35 : [1 , 34 , "y"], 36 : [1 , 34 , "y"] , 36 : [1 , 34 , "y"] , 37 : [1 , 34 , "y"],  38 : [1 , 34 , "y"], 39 : [2 , 39 , "n"] , 40 : [2 , 39 , "n"] , 41 : [2 , 39 , "n"] , 42 : [2 , 39 , "n"], 43 : [2 , 39 , "n"]}
child_with_res_blocks = {4 , 5 , 6 , 7}

def pi_latency(x1  , family_resnet , model , c_edge , s_edge):
    """
    x1 : Layers on edge (0 to total_network_layers - 1)
    
    family_resnet : True/False (True is resnet family else false) (Currently supports resnet18)
    (Support for other resnet models will be added later)
    
    model : example - resnet18 = models.resnet18(pretrained=True) 
    Pass the above in function , Resnet18 has total 45 layers (not including downsampling)

    c_edge , s_edge , c_server , s_server : Used in previous papers (GGhz)

    bandwidth = mbps
    image_path : Path to image of imagenet dataset because need to calculate upload latency
    
    Returns in seconds
    """    
    if family_resnet == False:
        children_list = get_children(model)
        edge_list = children_list[:x1+1]
        total_size_edge = 0
        for layer in edge_list:            
            total_size_edge += get_layer_parameter(layer)
        
        total_latency = get_pi_latency(total_size_edge , c_edge , s_edge )
        return total_latency


    if family_resnet == True:   

            resnet_child_list = list(model.children())
            if child_dict[x1] not in child_with_res_blocks:
                edge_list = resnet_child_list[0 : child_dict[x1] + 1]

                total_size_edge = 0
                for layer in edge_list:
                    total_size_edge += get_layer_parameter(layer)                 

                
                total_latency = get_pi_latency(total_size_edge , c_edge , s_edge)
                return total_latency
            
            else:
                child_number = child_dict[x1]
                previous_child_list = resnet_child_list[0:child_number]
                total_size_edge = 0
                for layer in previous_child_list:
                    total_size_edge += get_layer_parameter(layer)

                layer_block_number = block_dict[x1][0]
                block_starting_layer = block_dict[x1][1]
                seq_module_block = resnet_child_list[child_number]
                seq_module_block_list = list(seq_module_block.children())


                if layer_block_number == 1:
                    previous_output = input_batch.clone()
                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1_list = list(seq_module_block_list[0].children())
                    layers_on_edge = block_1_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        total_size_edge += get_layer_parameter(layer)
                    
                else:

                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1= seq_module_block_list[0]
                                    
                    total_size_edge += get_layer_parameter(block_1)

                    previous_output = input_batch.clone()
                    block_2_list = list(seq_module_block_list[1].children())    
                    layers_on_edge = block_2_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        total_size_edge += get_layer_parameter(layer)
                        
                total_latency = get_pi_latency(total_size_edge , c_edge , s_edge )
                return total_latency



def upload_latency(x1  , family_resnet , model , bandwidth , input_batch):
    """
    x1 : Layers on edge (0 to total_network_layers - 1)
    
    family_resnet : True/False (True is resnet family else false) (Currently supports resnet18)
    (Support for other resnet models will be added later)
    
    model : example - resnet18 = models.resnet18(pretrained=True) 
    Pass the above in function , Resnet18 has total 45 layers (not including downsampling)

    c_edge , s_edge , c_server , s_server : Used in previous papers (GGhz)

    bandwidth = mbps
    image_path : Path to image of imagenet dataset because need to calculate upload latency
    
    Returns in seconds
    """    
    if family_resnet == False:
        children_list = get_children(model)
        edge_list = children_list[:x1+1]
        server_list = children_list[x1+1:]
        for layer in edge_list:            
            with torch.no_grad():
                try:
                    input_batch = layer(input_batch)
                except:
                    input_batch = input_batch.view(1 , -1)
                    input_batch = layer(input_batch)
        
        
        activation_size = get_activation_size(input_batch)
        total_latency = get_upload_latency(activation_size , bandwidth)
        return total_latency


    if family_resnet == True:   

            resnet_child_list = list(model.children())
            if child_dict[x1] not in child_with_res_blocks:
                edge_list = resnet_child_list[0 : child_dict[x1] + 1]

                for layer in edge_list:
                    with torch.no_grad():
                        input_batch = layer(input_batch)
                
                
                activation_size = get_activation_size(input_batch)
                total_latency = get_upload_latency( activation_size , bandwidth )
                return total_latency
            
            else:
                child_number = child_dict[x1]
                previous_child_list = resnet_child_list[0:child_number]
                for layer in previous_child_list:
                    with torch.no_grad():
                        input_batch = layer(input_batch)

                layer_block_number = block_dict[x1][0]
                block_starting_layer = block_dict[x1][1]
                seq_module_block = resnet_child_list[child_number]
                seq_module_block_list = list(seq_module_block.children())


                if layer_block_number == 1:
                    previous_output = input_batch.clone()
                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1_list = list(seq_module_block_list[0].children())
                    layers_on_edge = block_1_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        with torch.no_grad():
                            input_batch = layer(input_batch)     

                else:

                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1= seq_module_block_list[0]
                    with torch.no_grad():
                        input_batch = block_1(input_batch)                    

                    previous_output = input_batch.clone()
                    block_2_list = list(seq_module_block_list[1].children())    
                    layers_on_edge = block_2_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        with torch.no_grad():
                            input_batch = layer(input_batch)              
                
                


                activation_size = get_activation_size_residual(input_batch , previous_output )
                total_latency = get_upload_latency( activation_size  , bandwidth )
                return total_latency




def total_latency(x1  , family_resnet , model , c_edge , s_edge , bandwidth , c_server , s_server , input_batch):
    """
    x1 : Layers on edge (0 to total_network_layers - 1)
    
    family_resnet : True/False (True is resnet family else false) (Currently supports resnet18)
    (Support for other resnet models will be added later)
    
    model : example - resnet18 = models.resnet18(pretrained=True) 
    Pass the above in function , Resnet18 has total 45 layers (not including downsampling)

    c_edge , s_edge , c_server , s_server : Used in previous papers (GGhz)

    bandwidth = mbps
    image_path : Path to image of imagenet dataset because need to calculate upload latency
    
    Returns in seconds
    """    
    if family_resnet == False:
        children_list = get_children(model)
        edge_list = children_list[:x1+1]
        server_list = children_list[x1+1:]
        total_size_edge = 0
        for layer in edge_list:            
            total_size_edge += get_layer_parameter(layer)
            
            with torch.no_grad():
                try:
                    input_batch = layer(input_batch)
                except:
                    input_batch = input_batch.view(1 , -1)
                    input_batch = layer(input_batch)
        total_size_server = 0
        for layer in server_list:            
            total_size_server += get_layer_parameter(layer)
        
        
        activation_size = get_activation_size(input_batch)
        total_latency = get_total_latency(total_size_edge , total_size_server , activation_size , c_edge , s_edge , bandwidth , c_server , s_server)
        return total_latency


    if family_resnet == True:   

            resnet_child_list = list(model.children())
            if child_dict[x1] not in child_with_res_blocks:
                edge_list = resnet_child_list[0 : child_dict[x1] + 1]

                total_size_edge = 0
                for layer in edge_list:
                    total_size_edge += get_layer_parameter(layer)
                    with torch.no_grad():
                        input_batch = layer(input_batch)
                
                server_list = resnet_child_list[child_dict[x1] + 1:]
                total_size_server = 0
                for layer in server_list:                    
                    total_size_server += get_layer_parameter(layer)
                
                activation_size = get_activation_size(input_batch)
                total_latency = get_total_latency(total_size_edge , total_size_server , activation_size , c_edge , s_edge , bandwidth , c_server , s_server)
                return total_latency
            
            else:
                child_number = child_dict[x1]
                previous_child_list = resnet_child_list[0:child_number]
                total_size_edge = 0
                for layer in previous_child_list:
                    total_size_edge += get_layer_parameter(layer)
                    with torch.no_grad():
                        input_batch = layer(input_batch)

                layer_block_number = block_dict[x1][0]
                block_starting_layer = block_dict[x1][1]
                seq_module_block = resnet_child_list[child_number]
                seq_module_block_list = list(seq_module_block.children())


                if layer_block_number == 1:
                    previous_output = input_batch.clone()
                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1_list = list(seq_module_block_list[0].children())
                    layers_on_edge = block_1_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        total_size_edge += get_layer_parameter(layer)
                        with torch.no_grad():
                            input_batch = layer(input_batch)     

                else:

                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1= seq_module_block_list[0]
                    with torch.no_grad():
                        input_batch = block_1(input_batch)                    
                    total_size_edge += get_layer_parameter(block_1)

                    previous_output = input_batch.clone()
                    block_2_list = list(seq_module_block_list[1].children())    
                    layers_on_edge = block_2_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                        total_size_edge += get_layer_parameter(layer)
                        with torch.no_grad():
                            input_batch = layer(input_batch)

                total_size_server = 0
                child = resnet_child_list[child_dict[x1]]
                children_of_child_list = list(child.children())
                block_number_of_layer = block_dict[x1][0]
                parent_block_of_layer = children_of_child_list[block_number_of_layer -1]
                parent_block_list = list(parent_block_of_layer.children())
                block_starting_layer = block_dict[x1][1]
                layer_difference_from_start_block = x1 - block_starting_layer
                if block_dict[x1][2] == "y":
                    block_layers_to_be_computed = parent_block_list[layer_difference_from_start_block + 1 : -1]
                    downsampling_layer = parent_block_list[-1]
                    for layer in block_layers_to_be_computed:                     
                        total_size_server += get_layer_parameter(layer)                   
                    total_size_server += get_layer_parameter(downsampling_layer)
                
                else:
                    block_layers_to_be_computed = parent_block_list[layer_difference_from_start_block + 1 :]
                    for layer in block_layers_to_be_computed:                        
                        total_size_server += get_layer_parameter(layer)
                
                if block_number_of_layer == 1:
                    second_block = children_of_child_list[1]                    
                    total_size_server += get_layer_parameter(second_block)
                
                next_list_of_child = resnet_child_list[child_dict[x1] + 1 : ]
                for child in next_list_of_child:                
                    total_size_server += get_layer_parameter(child)


                activation_size = get_activation_size_residual(input_batch , previous_output )
                total_latency = get_total_latency(total_size_edge , total_size_server , activation_size , c_edge , s_edge , bandwidth , c_server , s_server)
                return total_latency


def memory(x1,family_resnet , model):
    """
    Return in Mb    
    """
    if family_resnet == False:
        children_list = get_children(model)
        edge_list = children_list[:x1+1]
        server_list = children_list[x1+1:]
        total_size_edge = 0
        for layer in edge_list:            
            total_size_edge += get_layer_parameter(layer)            
        return total_size_edge / (10**6)
    if family_resnet == True:            
            resnet_child_list = list(model.children())
            if child_dict[x1] not in child_with_res_blocks:
                edge_list = resnet_child_list[0 : child_dict[x1] + 1]
                total_size_edge = 0
                for layer in edge_list:                    
                    total_size_edge += get_layer_parameter(layer)
                return total_size_edge / (10**6)
                
            else:
                child_number = child_dict[x1]
                previous_child_list = resnet_child_list[0:child_number]
                total_size_edge = 0
                for layer in previous_child_list:                  
                    total_size_edge += get_layer_parameter(layer)

                layer_block_number = block_dict[x1][0]
                block_starting_layer = block_dict[x1][1]
                seq_module_block = resnet_child_list[child_number]
                seq_module_block_list = list(seq_module_block.children())

                if layer_block_number == 1:
                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1_list = list(seq_module_block_list[0].children())
                    layers_on_edge = block_1_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                                
                        total_size_edge += get_layer_parameter(layer)
                else:
                    layer_difference_from_start_block = x1 - block_starting_layer
                    block_1= seq_module_block_list[0]
                                      
                    total_size_edge += get_layer_parameter(block_1)
                    
                    block_2_list = list(seq_module_block_list[1].children())    
                    layers_on_edge = block_2_list[0 : layer_difference_from_start_block + 1]
                    for layer in layers_on_edge:
                                           
                        total_size_edge += get_layer_parameter(layer)
                return total_size_edge / (10**6)





input_batch = open_image("E:\JournalPaper\leopard.jpg")


layer_arr = []
total_latency_arr = []
upload_latency_arr = []
memory_arr = []
pi_latency_arr = []


family_resnet =  False
memory_size = 4000
c_edge = 2
s_edge = 1.5
c_server = 6
s_server = 2.6
bandwidth = 10
input_batch = open_image("E:\JournalPaper\leopard.jpg")
model = models.vgg11(pretrained=True)

model_children = get_children(model)
layers = len(model_children)

for i in range(0 , layers):
    layer_arr.append(i)
    total_latency_arr.append(total_latency(i , family_resnet , model , c_edge , s_edge , bandwidth , c_server , s_server , input_batch))
    upload_latency_arr.append(upload_latency(i , family_resnet , model , bandwidth , input_batch))
    pi_latency_arr.append(pi_latency(i , family_resnet , model , c_edge , s_edge))
    #memory_arr.append(memory(i , family_resnet , model))
    print(i)
    #print(upload_latency_arr)
    

import matplotlib.pyplot as plt 


plt.title("Latency VGG11(in Sec)")

plt.xlabel("Split Index")
plt.ylabel("Latency(in Sec)")
plt.grid()
plt.plot(layer_arr , total_latency_arr , label = "Total Latency")
plt.plot(layer_arr , upload_latency_arr , label = "Upload Latency")
plt.plot(layer_arr , pi_latency_arr , label = "Edge Latency")
plt.legend()
plt.show()
