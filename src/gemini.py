import google.generativeai as genai

genai.configure(api_key="AIzaSyDJJi0munxtv0jOPPukCeXcQ2t0XM1FNAE")

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = "Hello Gemini! Can you confirm connection?"
response = model.generate_content(prompt)
print(response.text)
