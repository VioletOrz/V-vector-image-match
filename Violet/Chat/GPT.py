from openai import OpenAI

['gpt-3.5-turbo', 'gpt-4', 'gpt-4-0613','text-davinci-003', 'text-davinci-004', 'gpt-4-1106-preview', 'gpt-4-32k', 'gpt-4o']
key = ""

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=key
)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "user", "content": "你好?"},

  ]
)
print(response.choices[0].message.content)

class GPT():
    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api2.aigcbest.top/v1",
        )
        self.messages = []

    def chat(self, system,contxt: str = ""):
       a = 1 

