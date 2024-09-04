import discord
import requests
from bs4 import BeautifulSoup
from discord.ext import commands
from dotenv import load_dotenv
from newspaper import Article
import os
import logging
from logging.handlers import RotatingFileHandler
import nltk

# Load environment variables
load_dotenv()
nltk.download('punkt_tab')

# Discord and HuggingFace tokens
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
CHANNEL_ID = os.getenv('CHANNEL_ID')

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_KEYPHRASE_URL = "https://api-inference.huggingface.co/models/ml6team/keyphrase-extraction-distilbert-inspec"
MAX_INPUT_LENGTH = 3000

# Logging configuration with rotation
log_handler = RotatingFileHandler(
    "bot_log.log",  
    maxBytes=5*1024*1024,  # Maximum file size (e.g., 5 MB)
    backupCount=2  
)

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,  # Rotating log handler
        logging.StreamHandler()  # Also logs to console
    ]
)

logger = logging.getLogger(__name__)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        full_text = article.text
        logger.info(f"Text extracted from URL: {url}")
        return full_text
    except Exception as e:
        logger.error(f"Error fetching the content from {url}: {str(e)}")
        return None, f"Error fetching the content: {str(e)}"

def summarize_with_huggingface(text_chunk):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text_chunk,
        "parameters": {"min_length": 100, "do_sample": False}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        logger.info("Summarization completed successfully.")
        return response.json()[0]['summary_text']
    else:
        logger.error(f"Error summarizing the text: {response.text}")
        return f"Error summarizing the text: {response.text}"

def recursive_summary(text):
    summaries = []
    for i in range(0, len(text), MAX_INPUT_LENGTH):
        text_chunk = text[i:i + MAX_INPUT_LENGTH]
        summary = summarize_with_huggingface(text_chunk)
        summaries.append(summary)
    return " ".join(summaries)

def extract_tags_with_huggingface(text, retries=3, wait_time=10, max_tags=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    for attempt in range(retries):
        response = requests.post(HF_KEYPHRASE_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            keyphrases = [phrase['word'] for phrase in result]
            logger.info("Tags extracted successfully.")
            return keyphrases[:max_tags]
        else:
            error_message = response.json().get("error", "Unknown error")

            if "loading" in error_message.lower():
                logger.warning(f"Model loading. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)  # Wait for some time before retrying
            else:
                logger.error(f"Error extracting tags: {response.text}")
                return [f"Error extracting tags: {response.text}"]

    logger.error("Model is still loading after multiple attempts. Please try again later.")
    return ["Error: Model is still loading after multiple attempts. Please try again later."]

@bot.event
async def on_ready():
    logger.info(f"Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if str(message.channel.id) == CHANNEL_ID and bot.user.mentioned_in(message) and "http" in message.content:
        url = None
        for word in message.content.split():
            if word.startswith("http"):
                url = word
                break
        
        if url:
            logger.info(f"Message received with URL: {url}")

            # Extract text from the URL
            text = extract_text_from_url(url)
            if isinstance(text, tuple) and "Error" in text[1]:
                logger.error(f"Error during text extraction: {text[1]}")
                await message.channel.send("Error fetching content.")
                return
            
            # Summarize the extracted text
            summary = recursive_summary(text)
            logger.info("Summarization completed.")

            # Send summary
            if len(summary) > 2000:
                truncated_summary = summary[:1800] + "..."
                await message.channel.send(f"**Summary (truncated):**\n{truncated_summary}\n[Read the full article here]({url})")
                await message.channel.send("Summary is too long, it's better to read for yourself.")
            else:
                await message.channel.send(f"**Summary:**\n{summary}\n[Read the full article here]({url})")

            # Extract and send tags
            tags = extract_tags_with_huggingface(summary)
            logger.info("Tags created.")
            await message.channel.send(f"**Tags:** {', '.join(tags)}")

    await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)
