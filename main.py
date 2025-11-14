# imports
import os
import logging
import signal
import asyncio
from dotenv import load_dotenv
from discord import Intents
from mydiscord import BeerBot


# load environment
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
    logging.getLogger().addHandler(handler)

    intents = Intents.default()
    intents.message_content = True

    bot = BeerBot(TOKEN, intents=intents)

    # Run the bot in an asyncio event loop, so we can handle signals gracefully
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop_loop_on_signal(signame):
        logging.info(f"Received signal {signame}; shutting down")
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _stop_loop_on_signal(s.name))
        except NotImplementedError:
            # Windows event loop policy may not support signal handlers for all signals
            pass

    try:
        bot.run()
    except Exception:
        logging.exception("Bot stopped due to exception")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())


if __name__ == '__main__':
    main()