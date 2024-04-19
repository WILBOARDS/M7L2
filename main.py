import discord.py 
import os 
from keras.models import load_model
from PIL import Image, ImageOps
from discord.ext import commands
import numpy as np

np.set_printoptions(suppress=True)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')
    
def get_class(image_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    label_path = "labels.txt"  # Tambahkan path lengkap jika diperlukan
    class_names = open(label_path, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load the image from the provided path
    image = Image.open(image_path).convert("RGB")  # Menggunakan path gambar yang diterima

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Mengembalikan hasil prediksi dan confidence score
    return f"Class: {class_name[2:]}, Confidence Score: {confidence_score}"

@bot.command()
async def save(ctx):
    if ctx.message.attachments:
        directory = 'Photos'
        
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        for attachment in ctx.message.attachments:
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                filename = os.path.join(directory, attachment.filename)  # Combine directory path with filename
                try:
                    await attachment.save(filename)
                    predicted_class = get_class(filename)
                    await ctx.send(f'Attachment saved to {filename}. Predicted class: {predicted_class}')
                except Exception as e:
                    await ctx.send(f'Error saving attachment: {e}')
            else:
                await ctx.send('Only PNG, JPG, and JPEG files are allowed.')
    else:
        await ctx.send('No attachment found.')

    # print(get_class(r"Photos\download.jpg"))

@bot.command()
async def heh(ctx, count_heh=5):
    await ctx.send("he" * count_heh)

bot.run("ODAxMzM3MDM0MDUxODEzMzc3.GS2WbI.UexbflvH2UYVP3ioryPKBFpCcK9WorPzGIXJlc")