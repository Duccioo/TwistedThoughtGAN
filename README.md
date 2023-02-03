

<img src="https://res.cloudinary.com/duccio-me/image/upload/c_scale,r_300000,w_200/v1675424077/output_11_qvbcgv.jpg" width=200 height=200 align="right">


# TwistedThoughtGAN: Generating Lissajous Figures
Code for the Neural Network exam.

This repository contains the code for a GAN neural network that aims to generate Lissajous figures. Although the results are not exactly Lissajous figures, they are unique and resemble mind flayer or upside down structures. The code is written in Python and includes various modifications to the network architecture, loss functions, learning rate, and other parameters. The purpose of this repository is to showcase the potential of GANs in generating creative and unexpected outputs, even when the original goal is not fully achieved.
## Environment Variables

To run the training phase, you will need to add the following environment variables to your .env file or you can use the default ones in the `/script/main.py`


`EPOCH = *default 100*`

`BATCH_SIZE= *default 100*`

`DISC_STEP = *number of iteration before training the generator*`

`LEARNING_RATE_GENERATOR = *default 0.0002*`

`LEARNING_RATE_DISCRIMINATOR *default 0.0002*`

`GRADIENT_PENALTY = *default 0.0*` 

`STEPS_PER_IMG_SAVE = *number of step before saving image and optional sand telegram message*`

- Optional if you want to send an alert on telegram: 

    `TELEGRAM_TOKEN = *your_telegram_bot_token*`

    `CHAT_ID= *your_chat_id_for_sanding_the_messages`
## Dataset

1) create your own Dataset using the script based on [@tuxar-uk](https://github.com/tuxar-uk/Harmonumpyplot) with:

on the project directory:
```bash
  python data/harmonograph.py
```
Here you can play with different options using this function:
```python
gen_harmonograph()
``` 


For exsample if you want to create 100 harmonograph with dimension 256x256 in the directory ```/out_img/``` you can use this call:

```python
gen_harmonograph(n_images=[0, 100], dimension=256, path="out_img")
``` 

2) you can download the Dataset from Google Drive:
[Google Drive Link](https://drive.google.com/drive/folders/1WtfWxq7GHd4kZtoEF4L3SIb9cJG9tKSh?usp=sharing)

## Run Locally

Clone the project

```bash
  git clone https://github.com/Duccioo/TwistedThoughtGAN
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Set the Dataset on the proper place in the folder ```script/input/``` and than in the correct directory.

- For 128x128 pixel images: ```script/input/128/*your_directory_with_inside_all_the_128x128_images*/*all_128x128images*```

- For 256x256 pixel images: ```script/input/256/*your_directory_with_inside_all_the_128x128_images*/*all_256x256images*```


Start the training

```bash
  python script/main.py *optional_command*
```


## Command

You can use several command to run the script.

- ```--mode *train or test or generate*``` :start the selected procedure (default: train)

    - If you want to use the ```--mode train``` you have to create inside ```script/``` an ```input/``` folder and inside ```input``` then create a ```128/``` folder and a ```256/``` folder, the check the [Run Locally Paragraph](https://github.com/Duccioo/TwistedThoughtGAN/blob/main/README.md#run-locally):
        

    - If you want to use the ```--mode generate``` you have to put the desired model in the ```model``` folder and rename it to ```model.pt```

- ```--out *folder_out_name*```: set the folder for the output of the training procedure (default: generate a directory 'OUTPUT')

- ```--img_dim *128 or 256*```: set the correct image size for the model (default: 128)

- ```--random_seed *a number* ```: set the seed of the training phase, if <= 0 set it random (default: 42)

- ```--device *cpu or cuda*```: manual set cpu or gpu if available (default: try to use gpu if available)

- ```--telegram```: if setted try to send a message on a specific telegram chat specified on the environment variables when image is saved during training

- ```--gif```: create a gif with all the generated image saved during the training procedure 
## COLAB

You can use a built-in script in Google Colab to run a training example of this project


[GOOGLE COLAB LINK](https://colab.research.google.com/drive/16KQd0E_Xf5a1uc1Fkp38f1yfYY9M2zT9?usp=sharing)
## Report
[Link to the Report](https://duccioo.github.io/TwistedThoughtGAN/TwistedThoughtGAN_report_v2.pdf)



## Feedback

If you have any feedback, please reach out to me at meconcelliduccio@gmail.com or visit my website 
[duccio.me](https://duccio.me )
## License

[MIT](https://choosealicense.com/licenses/mit/)

