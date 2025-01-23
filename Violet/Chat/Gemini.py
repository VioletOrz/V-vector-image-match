import google.generativeai as genai
import PIL
import time


class Gemini():
    def __init__(self, model_name = "gemini-2.0-flash-exp", api_key = None):
        model_list = ['gemini-2.0-flash-exp', 'gemini-1.5-flash','gemini-2.0-flash-exp-image']
        if model_name not in model_list:
            raise ValueError(f"model_name:{model_name} is not in {model_list}")
        
        if api_key is None:
            raise ValueError("api_key is None")
        
        self.model_name = model_name
        genai.configure(api_key=api_key, transport='rest')
        self.model = genai.GenerativeModel(self.model_name)
        self.history = []

    def clean_history(self):
        self.history = []
    
    def chat(self, system:str = None, context: str = None, img = None, img_path = None, video_path = None, history = None):

        messages = []
        is_context = False
        if img_path is not None:
            if type(img_path) is list:
                messages.append(context)
                is_context = True
                for p in img_path:
                    messages.append(PIL.Image.open(p))
            elif type(img_path) is str:
                messages.append(PIL.Image.open(img_path))
                messages.append(context)
                is_context = True
            else:
                raise ValueError("img_path type is not supported")
            
        if img is not None:
            if type(img) is list:
                for i in img:
                    messages.append(i)
            else: messages.append(img)
                
        
        if video_path is not None:
            print(f"Uploading file...")
            video_file = genai.upload_file(path=video_file_name)
            print(f"Completed upload: {video_file.uri}")

            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            messages.append(video_file)

        if is_context == False:
            messages.append(context)

        if history == None:
            chat = self.model.start_chat(
                history=self.history
            )
        else:
            chat = self.model.start_chat(
                history=history
            )

      
        try:
            response = chat.send_message(messages)
            self.history = chat.history
            return response.text
        except Exception as e:
            print(f"Error:{e}\n请重试")
            return e
                
    def call(self, system:str = None, context: str = None, img = None, img_path = None, video_path = None):
        messages = []
        is_context = False
        if img_path is not None:
            if img_path is list:
                messages.append(context)
                is_context = True
                for p in img_path:
                    messages.append(PIL.Image.open(p))
            elif img_path is str:
                messages.append(PIL.Image.open(img_path))
                messages.append(context)
                is_context = True
            else:
                raise ValueError("img_path type is not supported")
            
        if img is not None:
            if img_path is list:
                for i in img:
                    messages.append(i)
            else: messages.append(img)
        
        if video_path is not None:
            print(f"Uploading file...")
            video_file = genai.upload_file(path=video_file_name)
            print(f"Completed upload: {video_file.uri}")

            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            messages.append(video_file)

        if is_context == False:
            messages.append(context)

        
        try:
            response = self.model.generate_content(messages)
            return response.text
        except Exception as e:
            print(f"Error:{e}\n请重试")
            return e
                
        
        
if __name__ == "__main__":
    
    key = ''

    """
    gemini = Gemini(api_key=key, model_name='gemini-2.0-flash-exp')   

    print(gemini.call(context="你好？"))
    print(gemini.chat(context="你好？"))
    print(gemini.chat(context="今天上海市的天气如何？"))
    print(gemini.chat(context="我的上个问题是什么？你回答了什么？"))
    """

    # Upload the video and print a confirmation.
    video_file_name = r'C:\Users\FM\Desktop\1.txt'#r"C:\Users\FM\Desktop\2.mp4"

    genai.configure(api_key=key, transport='rest')

    #print(f"Uploading file...")
    #video_file = genai.upload_file(path=video_file_name)
    #print(f"Completed upload: {video_file.uri}")
    """
    import time

    # Check whether the file is ready to be used.
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)"""
    
    img = PIL.Image.open(r"C:\Users\FM\Desktop\QQ截图20241219180829.png")

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    """chat = model.start_chat(
        history=[
            {"role": "user", "parts": "你好？"},
            {"role": "model", "parts": "你好，很高兴认识你"},
        ]
    )"""
    prompt = "请你根据这个视频的内容，生成一些符合人类日常表述习惯的网络用语弹幕，每一秒的视频要生成十条对应的弹幕"
    #prompt = "请你根据这个图片的内容，生成一些符合人类日常表述习惯的网络用语弹幕，生成十条弹幕"
    prompt = "提取一下这张图最下方的字幕"
    #response = model.generate_content("我的上个问题是什么，你回答了什么？")

    response = model.generate_content([img, prompt], request_options={"timeout": 600})
    print(response.text)
    #response = chat.send_message("请问你是？")
    #print(response.text)
    #print(chat.history)
    #a = 1
    #chat = model.start_chat(
    #    history=chat.history
    #)
    #response = chat.send_message("我的上个问题是什么，你回答了什么？")
    #print(response.text)
    #print(chat.history)

    a = 1