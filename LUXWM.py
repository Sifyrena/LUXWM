from PIL import Image, ImageDraw, ImageFont, ExifTags, ImageOps
from colorthief import ColorThief
import blend_modes as bm
import numpy as np
from datetime import datetime
import pandas as pd
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18, 'font.family': "Avenir Next"})

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os
from os import listdir


def add_border(
        imagepath,
        Label=[],
        Social = True,
        watermark_ratio=1,
        Adaptive=True,  # If Adaptive, expand border for image watermarks. If not, tall images will be watermarked sideways
        watermark_ratio_max=0.3,  # Times Image Max Dimension
        Shuffle_Dominant=True,
        Shuffle_Options=4,
        Pad_Extra=True,
        padding_ratio=0.01):

    image = Image.open(imagepath)

    width, height = image.size
    
    if Social and (width > 3000 or height > 3000):

        print("Resizing for social media.")

        if height > width:
            ratio = width / height
            height = 3000
            width = int(height * ratio)

        else:
            ratio = height / width
            width = 3000
            height = int(width * ratio)

        image = image.resize((width, height))

    width, height = image.size
    #print(width, height)

    color_thief = ColorThief(imagepath)
    # get the dominant color

    if Shuffle_Dominant:

        colors = color_thief.get_palette(color_count=Shuffle_Options, quality=500)

        dominant_color = colors[np.random.randint(Shuffle_Options)]

    else:

        dominant_color = color_thief.get_color(quality=500)

    dom_R, dom_G, dom_B = dominant_color

    if (dom_R * 0.299 + dom_G * 0.587 + dom_B * 0.114) > 150:  #186:
        print("Light")
        
        try:
            Watermark = Image.open("./Private/_MarkD.png")
        except:
            Watermark = Image.new('RGB', (200, 200), (0, 0, 0))
        Method = bm.multiply

        Light = True

    else:
        print("Dark")
        try:
            Watermark = Image.open("./Private/_MarkW.png")
        except:
            Watermark = Image.new('RGB', (200, 200), (255, 255, 255))
        Method = bm.screen

        Light = False

    # Go!
    dim = max(height, width)
    wmsize = dim / 4
    if width / height >= 2:
        offset = (0, (dim - height) / 2)
        wmoffset = (0, dim - wmsize)
        infooffset = (dim - 2 * wmsize, dim - wmsize)

    else:
        dim = height + wmsize * 2  #
        offset = ((dim - width) / 2, wmsize)

        if width / height >= 3 / 4:

            wmoffset = ((dim - width) / 2, wmsize + height)
            infooffset = ((dim + width) / 2 - 2 * wmsize, wmsize + height)

        else:
            Pad_Extra = False
            wmsizeOld = wmsize
            wmsize = width / 3
            wmoffset = ((dim - width) / 2, wmsizeOld + height)
            infooffset = ((dim - width) / 2 + wmsize, wmsizeOld + height)

    # Convert to Integers
    dim = int(dim)
    wmsize = int(wmsize)
    offset = tuple(int(item) for item in offset)
    wmoffset = tuple(int(item) for item in wmoffset)
    infooffset = tuple(int(item) for item in infooffset)

    # Create Blank
    new_image = Image.new("RGB", (dim, dim), dominant_color)
    new_image.paste(image, offset)

    print("Watermark Size", wmsize)

    Watermark = Watermark.resize((wmsize, wmsize))
    new_image.paste(Watermark, wmoffset, Watermark)
    if Label != []:
        LabelImage = Image.new("RGB", (dim, dim), (0, 0, 0))

        Lwidth, Lheight = LabelImage.size

        Lheight = wmsize
        Lwidth = 2 * wmsize

        Label = Label.resize((Lwidth, Lheight))

        LabelImage.paste(Label, infooffset)

        if Light:
            LabelImage = ImageOps.invert(LabelImage)

        alpha = Image.new('L', new_image.size, 255)

        new_image.putalpha(alpha)
        #new_image.show()
        LabelImage.putalpha(alpha)
        #LabelImage.show()

        background_img = np.array(new_image).astype(float)
        foreground_img = np.array(LabelImage).astype(float)

        # Blend images
        opacity = 1  # The opacity of the foreground that is blended onto the background is 70 %.
        blended_img_float = Method(background_img, foreground_img, opacity)

        # Convert blended image back into PIL image
        blended_img = np.uint8(
            blended_img_float
        )  # Image needs to be converted back to uint8 type for PIL handling.
        new_image = Image.fromarray(
            blended_img
        )  # Note that alpha channels are displayed in black by PIL by default.
        # This behavior is difficult to change (although possible).
        # If you have alpha channels in your images, then you should give
        # OpenCV a try.

    if Pad_Extra:

        dimADL = int(dim * padding_ratio)

        final_image = Image.new("RGB",
                                (dim + dimADL + dimADL, dim + dimADL + dimADL),
                                dominant_color)
        final_image.paste(new_image, (dimADL, dimADL))

        return final_image

    else:

        return new_image
    
def exif_to_table(image_path, Quiet = True, Message = ''):
    # Open image
    image = Image.open(image_path)
    # Retrieve EXIF data

    exif_data = image._getexif()

    # Create a dictionary to store the EXIF data
    exif = {}

    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        exif[tag] = value

    # Extract the relevant EXIF data
    
    if not Quiet:
        title = input("Enter Title Here") or 'Untitled'
    else:
        title = Message

    # Get the date and time from the EXIF data
    try:
        date_time_str = exif_data[0x9003]
        timezone_offset = exif_data.get(0x882a)

        print(timezone_offset)

        if timezone_offset is not None:
            print("Timezone Offset Detected!")

        # Convert the date and time string to a datetime object
        date_time = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')

        # Format the date and time in the desired format
        formatted_date_time = date_time.strftime('%Y:%m:%d %H:%M')
    except:
        formatted_date_time = ""

    print(formatted_date_time)
    camera = exif.get('Model', 'N/A')
    aperture = exif.get('FNumber', 'N/A')
    iso = exif.get('ISOSpeedRatings', 'N/A')
    shutter_speed = exif.get('ExposureTime', 'N/A')

    if shutter_speed != "N/A":
        shutter_speed = float(shutter_speed)

        if shutter_speed < 1:

            shutter_speed = f"1/{np.round(1/shutter_speed,0):.0f}"

        else:
            shutter_speed = f"{shutter_speed:.1f}"

    focal_length = exif.get('FocalLength', 'N/A')
    # Create a dataframe to store the data
    data = {
        "": [title],
        'Time': [formatted_date_time],
        'Camera': [camera],
        'Aperture': [f"f/{aperture}"],
        'ISO': [iso],
        'Shutter speed': [f"{shutter_speed}s"],
        'Focal length': [f"{focal_length}mm"]
    }
    df = pd.DataFrame(data)
    return df


def print_table_as_image(tables, color=(0, 0, 0)):
    fig, axs = plt.subplots(3, 1, facecolor=color, dpi=300, figsize=(16, 8))

    for ax, table in zip(axs, tables):

        #print(ax)
        ax.axis('off')
        ytable = ax.table(edges='vertical',
                          cellText=table.values,
                          colLabels=table.columns,
                          loc='center',
                          cellLoc='center')
        ytable.set_fontsize(28)
        ytable.scale(1, 4)
        ytable_cells = ytable.properties()['children']
        for cell in ytable_cells:
            cell.get_text().set_fontsize(50)
            cell.get_text().set_color('white')

    #plt.show()
    return fig, ax


def figure_to_image(figure):
    canvas = FigureCanvas(figure)

    s, (width, height) = canvas.print_to_buffer()

    return Image.frombytes("RGBA", (width, height), s)

def Exif_to_Digest(imagepath, Quiet, Message):

    df = exif_to_table(imagepath, Quiet = Quiet, Message = Message)

    dfr1 = df[[""]]
    dfr2 = df[["Time", "Camera"]]
    dfr3 = df[["Aperture", "ISO", "Shutter speed", "Focal length"]]

    fig, ax = print_table_as_image([dfr1, dfr2, dfr3])

    return figure_to_image(fig)

def Process_Batch(folder,
                  Social=True,
                  Threshold=3000,
                  Quiet=True,
                  SeriesTitle="",
                  Idempotence=True):

    for imagefile in os.listdir(folder):

        if Idempotence:
            if (imagefile.startswith("Processed")):
                continue

        if (imagefile.endswith(".png") or imagefile.endswith(".PNG")
                or imagefile.endswith(".JPG") or imagefile.endswith(".jpg")
                or imagefile.endswith(".jpeg") or imagefile.endswith(".JPEG")):

            print(imagefile)
            image = Image.open(folder + "/" + imagefile)

            Label = Exif_to_Digest(folder + "/" + imagefile, Quiet, SeriesTitle)

            width, height = image.size

            image = add_border(folder + "/" + imagefile, Label, Social)

            image = image.convert('RGB')

            image.save(f"{folder}/Processed_{imagefile}")