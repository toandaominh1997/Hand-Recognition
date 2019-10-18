import torch 

def accuracy_score(output, target):
    correct = 0 
    total = 0 
    with torch.no_grad():
        correct = 0
        total = 0
    
    return correct/total
