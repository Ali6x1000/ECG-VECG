import torch 
import torchvision
import torchvision.transforms


# get our device ready for training 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ploting 224 x 224 images


# VCG derivation



# Preprosessing 

# Normalize the dataset (find stdr deviation and mean for the each channel)


# Datset loading




#  Model Architecture 

# fix model's first convultional layer as the first one is used to normal RGB

# training Logic 



