import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
from PIL import Image
from Detection import detection

# Class names and colors (BGR format for OpenCV)
class_names = ['bg', 'Pistol']
color_sample = [
    (0,0,0),       
    (100, 0, 0),    # WBC - Darker Red (More Brownish Tone)
]

# Telegram Bot Token (replace with your own)
TOKEN = ""

# Object Detection Function
def object_detection_image(image):
    image_with_boxes = detection(image, class_names, color_sample)
    # # Convert to NumPy array
    # image = np.array(image_with_boxes)
    _, img_encoded = cv2.imencode('.jpg', image_with_boxes)
    return BytesIO(img_encoded.tobytes())


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🔫 **Welcome to the Pistol Detection Bot!** 🎯\n\n"
        "📸 **Send me an image**, and I will:\n"
        "✅ Detect and classify pistols\n"
        "✅ Identify potential threats in images\n"
        "✅ Generate an image with bounding boxes 🔍\n\n"
        "**Detected Objects:**\n"
        "🟡 **Pistol**\n"
        "⚠️ **Potential Threat Warning**\n\n"
        "🚀 *Send an image to start the detection!*"
    )


async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🔫 **Pistol Detection Bot Help** 🎯\n\n"
        "This bot detects pistols in images and identifies potential threats. 📸\n"
        "Simply send an image, and I'll analyze it to detect:\n"
        "🟡 **Pistols**\n"
        "⚠️ **Potential Threat Alerts**\n\n"
        "🔍 The bot will also **generate an image with bounding boxes** to highlight detected pistols.\n\n"
        "**Available Commands:**\n"
        "📌 /start - Welcome message\n"
        "📌 /help - Information on how to use the bot\n"
        "📌 /about - Details about the detection model\n"
    )
    await update.message.reply_text(help_text)



# # About Command
async def about_command(update: Update, context: CallbackContext):
    about_text = (
        "🔫 **Pistol Object Detection Bot** 🎯\n\n"
        "This bot detects and classifies pistols in images using an advanced deep learning model. 🏆\n\n"
        
        "**🔍 Features:**\n"
        "✅ Detects and classifies **pistols** in images 🔍\n"
        "✅ **Bounding Boxes** around detected weapons for precise localization 📏\n"
        "✅ **Threat Identification**: Helps in security applications 🚨\n"
        "✅ **Deep Learning Model**: Utilizes **Faster R-CNN with ResNet-50** for high-accuracy detection 🧠\n"
        "✅ **Optimized Training Strategy**: Trained with SGD optimizer, StepLR learning rate scheduler, and augmentation 🚀\n"
        "✅ **Hardware Acceleration**: Runs efficiently on GPU for real-time performance ⚡\n\n"

        "**💡 Interesting Facts About Pistols:**\n"
        "🔹 **The first semi-automatic pistol** was invented in 1893 by **Hugo Borchardt**. 🏛️\n"
        "🔹 **The Glock 17** is one of the most widely used pistols by law enforcement worldwide. 👮\n"
        "🔹 **Revolvers vs. Semi-Automatic Pistols**: Revolvers use a rotating cylinder, while semi-autos use a magazine for faster reloading. 🔄\n"
        "🔹 **The world's smallest pistol**, the **SwissMiniGun**, is only 5.5 cm long but fires real bullets! 🔬\n"
        
        "📸 *Send an image to detect pistols now!*"
    )
    await update.message.reply_text(about_text)


# Handle Photo Messages
async def handle_photo(update: Update, context: CallbackContext):
        photo = update.message.photo[-1:][0]  # Process the highest resolution photo
        file = await photo.get_file()
        image_bytes = BytesIO(await file.download_as_bytearray())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert NumPy array to PIL image
        image = Image.fromarray(image)
        
        image_with_boxes= object_detection_image(image)

        # Send Detected image with boxes result
        image_with_boxes.seek(0)  # Reset file pointer before sending    
        await update.message.reply_photo(photo=image_with_boxes, caption=f"Weapon Detcted Image")


# Main Function
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
