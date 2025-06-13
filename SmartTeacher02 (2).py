import cv2
from deepface import DeepFace
import pyaudio
import wave
from gtts import gTTS
import pygame
import io
import openai
import assemblyai as aai
import os
import mediapipe as mp

# Global variables to store gender and age
saved_gender = ""
saved_age = ""

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def analyze_face(frame):
    global saved_gender, saved_age

    # Detect faces in the frame using Haar cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No face detected", "N/A"

    # Perform face analysis using DeepFace
    result = DeepFace.analyze(frame, actions=["gender", "age"], enforce_detection=False)

    if isinstance(result, list) and len(result) > 0:
        # Access the first element of the result list
        face_analysis = result[0]

        # Extract gender with the highest probability
        gender_probabilities = face_analysis["gender"]
        gender = max(gender_probabilities, key=gender_probabilities.get)

        # Extract age
        age = face_analysis["age"]

        # Update global variables
        saved_gender = gender
        saved_age = age

        return gender, age
    else:
        return "No face detected", "N/A"


def check_distance(frame):
    # Calculate the distance between your face and the camera
    face_size_threshold = 30000  # Adjust this value based on your setup
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        face = faces[0]
        face_size = face[2] * face[3]
        if face_size > face_size_threshold:
            return "Yes"
        else:
            return "No"
    else:
        return "Yes"


def detect_hand_waving(frame):
    # Use MediaPipe Hands to detect hand gestures
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Check for waving gesture
            # This is a simple check for significant movement in the y-axis (up-down movement)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            if abs(thumb_tip - pinky_tip) > 0.1:  # Adjust this threshold based on your setup
                return True
    return False


def main():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()

    while True:
        ret, frame = cap.read()
        gender, age = analyze_face(frame)
        close = check_distance(frame)
        waving = detect_hand_waving(frame)
        print(f"Gender: {gender}, Age: {age}, Waving: {waving}, Close: {close}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Gender: {gender}, Age: {age}, Waving: {waving}, Close: {close}',
                    (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if close == "Yes" and waving:
            # Set up OpenAI API key
            api_key = " # APi-key"
            openai.api_key = api_key
            aai.settings.api_key = "cc6f58e0f3904f6bb1c281c0d296336a"

            def record_audio(filename, duration=10, chunk=1024, channels=1, sample_rate=44100):
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16,
                                    channels=channels,
                                    rate=sample_rate,
                                    input=True,
                                    frames_per_buffer=chunk)
                frames = []
                print("Recording...")
                for i in range(0, int(sample_rate / chunk * duration)):
                    data = stream.read(chunk)
                    frames.append(data)
                print("Finished recording.")
                stream.stop_stream()
                stream.close()
                audio.terminate()
                wf = wave.open(filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
                wf.close()

            def transcribe_audio(filename):
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(filename)
                if transcript.status == aai.TranscriptStatus.error:
                    print(transcript.error)
                    return None
                else:
                    return transcript.text

            def send_message(message_log):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
                    messages=message_log,  # The conversation history up to this point, as a list of dictionaries
                    max_tokens=3800,  # The maximum number of tokens (words or subwords) in the generated response
                    stop=None,  # The stopping sequence for the generated response, if any (not used here)
                    temperature=0.7,  # The "creativity" of the generated response (higher temperature = more creative)
                )
                for choice in response.choices:
                    if "text" in choice:
                        return choice.text
                return response.choices[0].message.content

            def main(saved_gender, saved_age):
                message_log = [{"role": "system", "content": "You are a helpful assistant."},
                               {"role": "system",
                                "content": f"Your saved gender is {saved_gender} and your saved age is {saved_age}."}]
                while True:
                    record_audio("user_input.wav")
                    user_input = transcribe_audio("user_input.wav")
                    if user_input == "stop":
                        break
                    if user_input is None:
                        print("Error transcribing user input.")
                        continue
                    message_log.append({"role": "user", "content": user_input})
                    response = send_message(message_log)
                    tts = gTTS(text=response, lang='en')
                    audio_stream = io.BytesIO()
                    tts.write_to_fp(audio_stream)
                    audio_stream.seek(0)
                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_stream)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    message_log.append({"role": "assistant", "content": response})
                    if user_input == "stop":
                        print("Goodbye!")
                        cap.release()
                        cv2.destroyAllWindows()
                        break

            if __name__ == "__main__":
                saved_gender = gender
                saved_age = age
                main(saved_gender, saved_age)

        cv2.imshow('Camera Feed', frame)
        prev_frame = frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#######################################
####################################
#################################
#############################
#########################
###################
#############
########
####
##
#