Quick setup (Windows PowerShell)

1. Install Python 3.12 (recommended) or create a conda env.
   - Using py launcher if multi versions installed:
     py -3.12 -m venv .venv

2. Activate venv (PowerShell):
   .\.venv\Scripts\Activate.ps1

3. Upgrade pip and install deps:
   python -m pip install --upgrade pip
   pip install -r requirements.txt

4. Create a .env file (copy .env template) and set DISCORD_BOT_TOKEN.

5. Run locally:
   python main.py

Docker (optional):
- Build:
  docker build -t beerbotplus .
- Run:
  docker run --env-file .env -v "${PWD}/discord.log:/app/discord.log" --restart unless-stopped --name beerbotplus -d beerbotplus:latest
