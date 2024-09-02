import discord
import requests
from bs4 import BeautifulSoup
from discord.ext import commands
from dotenv import load_dotenv
from newspaper import Article
import os

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
CHANNEL_ID = os.getenv('CHANNEL_ID')

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

MAX_INPUT_LENGTH = 3000

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# kw_model = KeyBERT()  # Initialize KeyBERT

def extract_text_from_url(url):
    # try:
    #     headers = {
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
    #     }
    #     response = requests.get(url, headers=headers)
    #     # print(response.text)
    #     soup = BeautifulSoup(response.text, 'html.parser')
        
    #     paragraphs = soup.find_all('p')
    #     text = " ".join([para.get_text() for para in paragraphs])
    #     return text
    # except Exception as e:
    #     return f"Error fetching the content: {str(e)}"
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        full_text = article.text
        print(full_text)
        return full_text
    except Exception as e:
        return None, f"Error fetching the content: {str(e)}"

def summarize_with_huggingface(text_chunk):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
 
    payload = {
        "inputs": text_chunk,
        "parameters": {"min_length": 100, "do_sample": False}
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

# Function to extract keywords/tags
# def extract_tags(summary):
#     tags = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
#     return [tag[0] for tag in tags]

HF_KEYPHRASE_URL = "https://api-inference.huggingface.co/models/ml6team/keyphrase-extraction-distilbert-inspec"

def extract_tags_with_huggingface(text, retries=3, wait_time=10, max_tags=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    for attempt in range(retries):
        response = requests.post(HF_KEYPHRASE_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            # Extract keyphrases from the model's output
            keyphrases = [phrase['word'] for phrase in result]
            return keyphrases[:max_tags]
        else:
            error_message = response.json().get("error", "Unknown error")
            
            if "loading" in error_message.lower():
                print(f"Model is still loading. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)  # Wait for some time before retrying
            else:
                return [f"Error extracting tags: {response.text}"]

    return ["Error: Model is still loading after multiple attempts. Please try again later."]

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if str(message.channel.id) == CHANNEL_ID and bot.user.mentioned_in(message) and "http" in message.content:
        url = None
        for word in message.content.split():
            if word.startswith("http"):
                url = word
                break
        
        if url:
            text = extract_text_from_url(url)
            # print(f"Text:\n {text}")
            # if "Error" in text:
            #     print(text)
            #     await message.channel.send("Error")
            # if True:
            summary = recursive_summary(text)
            
            if len(summary) > 2000:
                truncated_summary = summary[:1800] + "..."
                await message.channel.send(f"**Summary (truncated):**\n{truncated_summary}\n[Read the full article here]({url})")
                await message.channel.send("Summary is too long, it's better to read for yourself.")
            else:
                await message.channel.send(f"**Summary:**\n{summary}\n[Read the full article here]({url})")
            
            # Extract and send tags
            tags = extract_tags_with_huggingface(summary)
            await message.channel.send(f"**Tags:** {', '.join(tags)}")

    await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)
