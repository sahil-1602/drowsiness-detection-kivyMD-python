from scipy.spatial import distance
import dlib
from imutils import face_utils
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
import cv2

class MainApp(MDApp):

    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)

        self.is_music_playing = False

        #Minimum threshold of eye aspect ratio below which alarm is triggerd
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.3

        #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 30

        #COunts no. of consecutuve frames below threshold value
        self.COUNTER = 0

        self.face_facade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

        #Load face detector and predictor, uses dlib shape predictor file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        #Extract indexes of facial landmarks for the left and right eye
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)

        return layout

    def play_music(self, *args):
        if (not self.is_music_playing):
            music = SoundLoader.load('audio/alert.wav')
            music.play()
            self.is_music_playing = True

    def load_video(self, *args):
        ret, frame = self.capture.read()
        # frame initialize
        self.image_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        face_rectangle = self.face_facade.detectMultiScale(gray, 1.3, 5)

        #Draw rectangle around each face detected
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Detect facial points
        for face in faces:

            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            #Get array of coordinates of leftEye and rightEye
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            #Calculate aspect ratio of both eyes
            leftEyeAspectRatio = self.eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = self.eye_aspect_ratio(rightEye)

            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            #Use hull to remove convex contour discrepencies and draw eye shape around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            #Detect if eye aspect ratio is less than threshold
            if(eyeAspectRatio < self.EYE_ASPECT_RATIO_THRESHOLD):
                self.COUNTER += 1
                #If no. of frames is greater than threshold frames,
                if self.COUNTER >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    self.play_music()
                    cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            else:
                self.is_music_playing = False
                self.COUNTER = 0

        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def take_picture(self, *args):
        image_name = 'picture_at_222.jpg'
        cv2.imwrite(image_name, self.image_frame)

    #This function calculates and return eye aspect ratio
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        ear = (A+B) / (2*C)
        return ear

if __name__ == '__main__':
    MainApp().run()