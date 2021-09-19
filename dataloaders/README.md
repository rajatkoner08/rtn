First train the SG algo if we want to incorporate temporal dependency.

In the dataloader there are two option:

1. Use Frame by Frame 
    a. Each scene graph are independent from each other.
    b. Generation of scene graph is dependent on prior grpah. In this case the output of 
       dataloader will preserve the sequence we want to give LSTM.
    
2. Use the whole video like I3D


Things to do :
1.Visualize the image and re shaped bb box(Done)
2.Run neural motif with vrd images.
3.run training then