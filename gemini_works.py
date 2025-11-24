import google.generativeai as genai
genai.configure(api_key="AIzaSyCB9kMkd7yiK5WuNMdTSoMt1oOxrsNMRnk")
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content("Say hello!")
print(response.text)