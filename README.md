# CS701_G8
Code base for the Medical Image Segmentation on 12 abdominal organs


<h2>To Do List: </h2>
<ul>
<li>Basic U-Net Architecture Code.
<li>DataLoader for the public dataset shared.
<li>Decide on U-Net Vs SAM.
<li><Initial Results of U-Net on the dataset.
<li>Check the pretrained u-net baest performing model.
</ul>

<h2>List of pretrained models </h2>
<ul>
<li>https://github.com/mberkay0/pretrained-backbones-unet/tree/main</li>
</ul>

<h2>Pointers to handle noisy images</h2> 

- For noisy images: Can change the loss funcation that is robut to handle noise (like Tversky loss)
- Denoise it using gaussian filters. 
  
<h2>Others:</h2> 

- GPU access and setup
