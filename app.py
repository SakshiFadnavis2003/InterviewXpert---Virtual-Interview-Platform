from flask import Flask, request, render_template, redirect, url_for, session
import os
import google.generativeai as genai
import logging
from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
import google.generativeai as genai
from werkzeug.utils import secure_filename
import logging


# Initialize Flask app
app = Flask(__name__)  # Required for using sessions

# Set up API key for generative model
os.environ["API_KEY"] = 'AIzaSyDY1U-9SQtnOrmCTBwpi5nrfARznuBUEzw'
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
logging.basicConfig(level=logging.DEBUG)
UPLOAD_FOLDER = r"C:\Users\tusha\OneDrive\Desktop\InterviewXpert\uploads"  # Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.secret_key = os.urandom(24)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        choice = request.form.get('choice')  # Get the user's choice from the form
        if choice == 'role':
            return redirect(url_for('index2', choice='role'))
        elif choice == 'cs':
            return redirect(url_for('index2', choice='cs'))
    return render_template('index.html')


# Function to generate questions based on input prompt
def generate_questions(prompt):
    try:
        response = model.generate_content(prompt)
        logging.info(f"Model response: {response.text}")
        questions = response.text.split("\n")

        filtered_questions = []
        for line in questions:
            if line.strip() and line[0].isdigit():  # Ensure line starts with a number
                filtered_questions.append(line.split(". ", 1)[-1].strip()) 

        # Ensure the first question is always "Introduce yourself"
        final_questions = ["Introduce yourself"] + filtered_questions[:9]  # Limit to 10 questions
        return final_questions
    except Exception as e:
        logging.error(f"Error in generating questions: {e}")
        return []

@app.route('/index2', methods=['GET', 'POST'])
def index2():
    choice = request.args.get('choice')  # Get the choice passed in the URL
    if choice == 'role':
        return render_template('index2.html', choice='role')
    elif choice == 'cs':
        return render_template('index2.html', choice='cs')
    return render_template('index2.html', choice=None)


# Route to handle role-based question generation
@app.route("/generate_role", methods=['GET', 'POST'])
def generate_role():
    if request.method == 'POST':
        role = request.form["role"]
        prompt = f"Generate 10 interview questions for a {role} role"
        questions = generate_questions(prompt)
        
        # Store the generated questions in session
        session['questions'] = questions
        
        return render_template("index2.html", questions=questions, role=role, choice="role")
    return render_template('index2.html')

# Route to handle CS topic-based question generation
@app.route("/generate_cs",methods=['GET', 'POST'])
def generate_cs():
    if request.method == 'POST':
        topic = request.form["topic"]
        prompt = f"Generate 10 questions on {topic}"
        questions = generate_questions(prompt)
        
        # Store the generated questions in session
        session['questions'] = questions
        
        return render_template("index2.html", questions=questions, topic=topic, choice="cs")
    return render_template('index2.html')

@app.route("/question/<int:question_id>", methods=["GET"])
def question_page(question_id=None):
    # Retrieve the list of questions from the session
    questions = session.get('questions', [])

    # Check if the question_id is within bounds and display the corresponding question
    if question_id is not None and question_id < len(questions):
        question = questions[question_id]
    else:
        question = "Question not found."
    
    return render_template("question.html", question=question)

def generate_feedback(transcription):
    prompt = f"Based on the following transcription of an interview response, provide constructive feedback:\n\nTranscription:\n{transcription}\n\nQuestion:\n{question}"
    response = model.generate_content(prompt)
    return response.text


def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

@app.route('/audio_to_text', methods=['POST'])
def audio_to_text():
    if 'audio' not in request.files:
        return "No audio file provided", 400

    audio_file = request.files['audio']
    
    # Check the file type
    print("Uploaded file type:", audio_file.content_type)

    # Save the uploaded audio file
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    try:
        # Convert audio to WAV if necessary
        if not audio_path.endswith('.wav'):
            wav_path = convert_to_wav(audio_path)  # Convert to WAV
        else:
            wav_path = audio_path

        # Transcribe the audio
        transcription = convert_audio_to_text(wav_path)
        return jsonify({"transcription": transcription})

    except Exception as e:
        print("Error during transcription:", str(e))
        return jsonify({"error": str(e)}), 500
    
def generate_feedback(transcription):
    # Feedback generation prompt
    prompt = f"""
    Please analyze the transcription of the interview for the candidate and make it in the readable format  and provide comprehensive feedback in the following format:

    1. Theoretical Feedback:
       - Assess the candidate's technical knowledge by examining their understanding of relevant concepts, their ability to explain technical terms, and their responses to technical questions as presented in the transcription.
       - Analyze their problem-solving skills by reviewing their approach to situational or technical questions, as well as their critical thinking and reasoning abilities demonstrated in their answers.
       - Highlight any strengths or areas for improvement in their responses.
       - Offer an overall impression of the candidate's suitability for the role, focusing on their technical and analytical abilities.

    2. Scoring Feedback (out of 10 for each aspect):
       - Technical Knowledge
       - Problem-Solving and Analytical Thinking
       - Overall Technical Suitability

    Transcription:
    {transcription}

    Ensure the feedback is clear, constructive, and provides actionable suggestions for improvement.
    """

    # Generate feedback using the generative model
    response = model.generate_content(prompt)
    return response.text


@app.route('/templates/feedback.html')
def feedback():
    return render_template('templates/feedback.html')

# Remove the redundant `generate_feedback()` function

@app.route('/submit', methods=['POST'])
def submit():
    transcription = request.form.get('transcription')
    feedback = generate_feedback(transcription)
    return jsonify({'transcription': transcription, 'feedback': feedback})


@app.route('/analyze_body_language', methods=['POST'])
def analyze_body_language_endpoint():
    video_file = request.files.get('video')
    if video_file:
        if video_file.content_type not in ['video/webm', 'video/mp4']:  # Validate file type
            return jsonify({"error": "Invalid video format. Please upload a WebM or MP4 file."}), 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        body_language_feedback = analyze_body_language(video_path)
        return jsonify(body_language_feedback)
    else:
        return jsonify({"error": "No video file provided."}), 400


# Initialize MediaPipe models
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_body_language(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file. Check if the file is valid."}
        
        feedback = {"face_touches": 0, "hand_gestures": []}
        overall_score = 100
        frame_counter = 0
        frame_skip = 5  # Process every 5th frame
        face_touch_last_frame = -10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            hand_results = hands.process(image)
            face_mesh_results = face_mesh.process(image)

            # Face Touch Analysis
            if face_mesh_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
                face_touch_detected = False
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for hand_landmark in hand_landmarks.landmark:
                            for face_landmark in face_landmarks.landmark:
                                distance = np.sqrt(
                                    (hand_landmark.x - face_landmark.x) ** 2 +
                                    (hand_landmark.y - face_landmark.y) ** 2
                                )
                                if distance < 0.05:
                                    face_touch_detected = True
                                    break

                    if face_touch_detected:
                        break

                if face_touch_detected and frame_counter - face_touch_last_frame > 10:
                    feedback["face_touches"] += 1
                    face_touch_last_frame = frame_counter
                    if len(feedback["hand_gestures"]) == 0:
                        feedback["hand_gestures"].append(
                            "Mistake: Face touch detected. "
                            "Correction: Try to avoid touching your face."
                        )
                    overall_score -= 5

        cap.release()
        cv2.destroyAllWindows()

        feedback["overall_score"] = max(0, int(overall_score))
        feedback["summary"] = (
            f"Overall Body Language Score: {feedback['overall_score']}%\n"
            f"Face touched: {feedback['face_touches']} times.\n"
            "Remember to avoid scratching or rubbing your face for a professional appearance.\n"
            f"Face Touch Count: {feedback['face_touches']}\n"
            f"Hand Gestures Feedback: {feedback['hand_gestures']}\n"
            f"Overall Score: {feedback['overall_score']}\n"
        )
        
        return feedback

    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {"error": str(e)}



if __name__ == "__main__":
    app.run(debug=True)
