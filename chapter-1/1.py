from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

context = """
...and it's Modrić on the ball in the midfield, looking up, spotting the run of Vinícius Júnior down the left flank. He threads a beautiful pass through the gap... Vini Júnior receives it, controls it perfectly! He's running at Araújo now, shifting it onto his right foot... a stepover! Can he get past? Araújo stands his ground... Vini Júnior cuts back inside... looking for a shooting angle... still on the ball, surrounded by two defenders... he tries to flick it through to Bellingham! Intercepted! A crucial tackle there from Gavi! Barcelona regain possession... they look to build from the back now...
"""


output = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role":"system", "content":"You are an expert commentator for football matches. Continue the live commentary provided by the user."},
    {"role":"user", "content":f"Here is the ongoing commentary for a Real Madrid vs Barcelona match. Please continue it naturally:\n\n{context}"},
    ],
    temperature=0.5,
    max_tokens=150,
    presence_penalty=0.6,
    frequency_penalty=0.5,
    logit_bias={1504: 5, 128: -100},
)

print(output.choices[0].message.content)