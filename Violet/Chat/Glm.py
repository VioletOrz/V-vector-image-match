###################################################
#
###################################################
from zhipuai import ZhipuAI


tools_web_search = [{
    "type": "web_search",
    "web_search": {
        "enable": True #默认为关闭状态（False） 禁用：False，启用：True。
    }
}]

class Glm(object):
    def __init__(self, model_name:str = 'glm-4-flash', api_key = None):
        model_list = 'glm-4-plus', 'glm-4-0520', 'glm-4', 'glm-4-air', 'glm-4-airx', 'glm-4-long', 'glm-4-flashx', 'glm-4-flash'
        if model_name not in model_list:
            raise ValueError(f"model_name:{model_name} is not in {model_list}")
        
        if api_key is None:
            raise ValueError("api_key is None")
        
        self.client = ZhipuAI(api_key = api_key)
        self.model_name = model_name
        self.message_history = []

    def chat(self, system:str = None, context: str = None, history = None):
        messages = []
        if system != None:
            messages.append({"role": "system", "content": system})
        if context != None:
            messages.append({"role": "user", "content": context})
        if history != None:
            messages.extend(history)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 填写需要调用的模型名称
                messages=messages,
                #tools=tools,
                #tool_choice={"type": "function", "function": {"name": "get_ticket_price"}},
            )
            text = response.choices[0].message.model_dump()['content']
            self.message_history.append(messages)
            self.message_history.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            print(e)

    def clean_msg_history(self):
        self.message_history = []
        

        