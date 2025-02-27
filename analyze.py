import pdfplumber
import os
import openai
import base64
import whisper
import cv2


openai.api_key = "your-api-key"


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    
    response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Analyze this PDF file text and provide insights:\n{text}"}
            ]
        )
    
    return response["choices"][0]["message"]["content"]


def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that extracts and analyzes text from images."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    }}]}
            ]
        )

        return response["choices"][0]["message"]["content"]



model = whisper.load_model("base")
def analyze_audio(audio_path):
    result = model.transcribe(audio_path)
    text = result["text"]
    response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Analyze this text and provide insights:\n{text}"}
            ]
        )
    
    return response["choices"][0]["message"]["content"]


def extract_frames(video_path, frame_interval=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


def analyze_video_frames(frames):
    insights = []
    for frame in frames:
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Analyze this video frame and provide insights:\n{frame}"}
            ]
        )
        insights.append(response['choices'][0]['message']['content'])
    
    return insights


file_name = "Koenigsegg.mp4"
file_path = os.path.join(os.path.dirname(__file__), f"data/{file_name}")

data = extract_frames(file_path)
print(data[:2])
