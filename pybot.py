import discord
import requests
from discord.ext import commands
from dotenv import load_dotenv
import os
from newspaper import Article
from keybert import KeyBERT
import nltk

nltk.download('punkt')  # Correct the download resource
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))  # Channel ID where bot will respond

print(f"DISCORD_BOT_TOKEN: {DISCORD_BOT_TOKEN}")
print(f"CHANNEL_ID: {CHANNEL_ID}")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

kw_model = KeyBERT()

def extract_text_and_summary_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        full_text = article.text
        print(full_text)
        summary = article.summary
        return full_text, summary
    except Exception as e:
        return None, f"Error fetching the content: {str(e)}"

def truncate_summary(summary, url):
    if len(summary) > 2000:
        return f"**Summary (truncated):**\n{summary[:1800]}...\n[Read the full article here]({url})\nSummary is too long, it's better to read for yourself."
    else:
        return f"**Summary:**\n{summary}\n[Read the full article here]({url})"

def extract_tags(summary):
    tags = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return [tag[0] for tag in tags]

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.channel.id == CHANNEL_ID and bot.user.mentioned_in(message) and "http" in message.content:
        url = None
        for word in message.content.split():
            if word.startswith("http"):
                url = word
                break
        
        if url:
            print(f"Processing URL: {url}")  # Debug: URL
            text, summary = extract_text_and_summary_from_url(url)
            if "Error" in summary:
                await message.channel.send(summary)
            else:
                truncated_message = truncate_summary(summary, url)
                tags = extract_tags(summary)
                tags_message = f"**Tags:** {', '.join(tags)}" if tags else "No tags found."
                await message.channel.send(f"{truncated_message}\n{tags_message}")
        else:
            print("No URL found in the message.")
    await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)
