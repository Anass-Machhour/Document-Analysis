import os
import time
from google import genai

apiKey = "AIzaSyAFXEHAl_tHoNdHLmhtXFp0ffrU0LsWRp0"
client = genai.Client(api_key=apiKey)

prompt = '''
    Provide a direct analysis and summary of the data without any introductions or 
    explanations. Extract key insights, summarize important points, and present 
    findings concisely. If applicable, interpret tables, graphs, images, or audio 
    transcriptions, highlighting essential details and patterns. Structure the response 
    clearly and omit any introductory or concluding remarks.
'''


def analyze_document(file_path):
    print("Uploading file...")
    file = client.files.upload(file=file_path)
    print(f"Completed upload: {file.uri}")

    while file.state.name == "PROCESSING":
        print('.', end='\n')
        time.sleep(1)
        file = client.files.get(name=file.name)

    if file.state.name == "FAILED":
        raise ValueError(file.state.name)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[file, prompt]
    )

    client.files.delete(name=file.name)
    return response.text



file_name = "handwriting.jpg"

file_path = os.path.join(os.path.dirname(__file__), f"data/{file_name}")

text = analyze_document(file_path)

print(text)
