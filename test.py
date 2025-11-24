import google.generativeai as genai

genai.configure(api_key="AIzaSyDJJi0munxtv0jOPPukCeXcQ2t0XM1FNAE")

for m in genai.list_models():
    print(m.name)
