import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something...")
    r.adjust_for_ambient_noise(source, duration=10.1)
    audio = r.listen(source)

text = r.recognize_google(audio)
print("You said:", text)

