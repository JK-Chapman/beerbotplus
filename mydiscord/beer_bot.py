from discord.ext import commands
import discord
import hashlib
import logging
from typing import Optional

class BeerBot:
    def __init__(self, token: str, *, intents: discord.Intents, command_prefix: str = "$"):
        self.token = token
        self.intents = intents
        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self._register_events()

    def _register_events(self):
        @self.bot.event
        async def on_ready():
            logging.info(f"Logged in as {self.bot.user} (id={self.bot.user.id})")

        # @self.bot.event
        # async def on_message(message: discord.Message):
        #     # ignore self messages
        #     if message.author == self.bot.user:
        #         return

        #     # simple command
        #     if message.content.startswith("$hello"):
        #         await message.channel.send("Hello!")

        #     # attachments: only handle the first attachment for now
        #     if message.attachments and message.content == "$cheers!".lower():
        #         attachment = message.attachments[0]
        #         if (attachment.content_type and attachment.content_type.startswith("image/")) or \
        #                 attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
        #             file_bytes = await attachment.read()
        #             checksum = hashlib.sha256(file_bytes).hexdigest()
        #             # TODO: validate checksum against DB and run model
        #             await message.channel.send(f"Received image (checksum {checksum[:8]}...)")
        #         else:
        #             await message.channel.send(f"Attachment {attachment.filename} is not an image.")

            # allow commands to be processed by commands extension
            # await self.bot.process_commands(message)

        @self.bot.command(name="cheers!".lower())
        async def cheers(ctx: commands.Context):
            message = ctx.message
            if message.attachments and message.content == "$cheers!".lower():
                attachment = message.attachments[0]
                if (attachment.content_type and attachment.content_type.startswith("image/")) or \
                    attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                    file_bytes = await attachment.read()
                    checksum = hashlib.sha256(file_bytes).hexdigest()
                    # TODO: validate checksum against DB and run model
                    await message.channel.send(f"Received image (checksum {checksum[:8]}...)")
                else:
                    await message.channel.send(f"Attachment {attachment.filename} is not an image.")
            else:
                await ctx.send("Please attach an image with the $cheers! command.")

        @self.bot.command(name="hello")
        async def hello(ctx: commands.Context):
            await ctx.message.channel.send("Hello!")

    def add_cog(self, cog: commands.Cog) -> None:
        """Add a Cog to the underlying bot."""
        self.bot.add_cog(cog)

    def run(self, *, log_handler: Optional[logging.Handler] = None) -> None:
        """Start the bot (blocking). Optionally attach a logging handler."""
        if log_handler:
            logging.getLogger().addHandler(log_handler)
        self.bot.run(self.token)

    async def close(self) -> None:
        """Async close helper."""
        await self.bot.close()