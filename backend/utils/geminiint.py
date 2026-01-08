# # gemini_helper.py
# import os
# import google.generativeai as genai

# # Initialize the Gemini client with API key from environment variable
# genai.configure(api_key=os.getenv("AIzaSyDuGvi3q4sunnnmSfwMryhtdPTJEvOoknI"))

# def explain_condition(predicted_condition: str) -> str:
#     prompt = f"""
#     You are a helpful medical assistant.
#     A patient has been diagnosed with {predicted_condition} based on symptoms.
#     1. Explain what {predicted_condition} is.
#     2. List common symptoms.
#     3. Provide appropriate precautions and prevention advice.
#     """
#     # Use GenerativeModel to generate content
#     model = genai.GenerativeModel("gemini-2.5-flash")
#     response = model.generate_content(
#         prompt
#     )

#     return response.text
