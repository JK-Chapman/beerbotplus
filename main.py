# Imports
import hashlib
import logging
import discord
import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()
token = os.environ.get('DISCORD_BOT_TOKEN')

# var setup
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

# Event Handling
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
    
    if message.content.attachments:
        attachment = message.attachments[0] # only support one attachment
        # Check if it's an image by content_type or extension
        if (attachment.content_type and attachment.content_type.startswith('image/')) or \
            attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # Download the attachment
            file_bytes = await attachment.read()
            # Calculate SHA256 checksum
            checksum = hashlib.sha256(file_bytes).hexdigest()
            # validate checksum hasn't been used before using postgresql

            # if valid then run the picture through the model

        else:
            await message.channel.send(f'Attachment {attachment.filename} is not an image.')


# Actually run the bot
client.run(token, log_handler=handler)