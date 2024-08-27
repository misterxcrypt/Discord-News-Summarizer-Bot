import discord
import requests
from bs4 import BeautifulSoup
from discord.ext import commands
from dotenv import load_dotenv
import os

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

MAX_INPUT_LENGTH = 5104

intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix="!", intents=intents)

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return f"Error fetching the content: {str(e)}"

def summarize_with_huggingface(text_chunk):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
 
    payload = {
        "inputs": text_chunk,
        "parameters": {"min_length": 50, "do_sample": False} 
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
   
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        return f"Error summarizing the text: {response.text}"

def recursive_summary(text):
    summaries = []
    for i in range(0, len(text), MAX_INPUT_LENGTH):
        text_chunk = text[i:i+MAX_INPUT_LENGTH] 
        summary = summarize_with_huggingface(text_chunk)
        summaries.append(summary)
    return " ".join(summaries) 

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if bot.user.mentioned_in(message) and "http" in message.content:
        url = None
        for word in message.content.split():
            if word.startswith("http"):
                url = word
                break
        
        if url:
            text = extract_text_from_url(url)
            if "Error" in text:
                await message.channel.send(text)
            else:
                summary = recursive_summary(text)
                
                if len(summary) > 2000:
                    truncated_summary = summary[:1800] + "..."
                    await message.channel.send(f"**Summary (truncated):**\n{truncated_summary}")
                    await message.channel.send("Summary is too long, it's better to read for yourself.")
                else:
                    await message.channel.send(f"**Summary:**\n{summary}")

    await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)
