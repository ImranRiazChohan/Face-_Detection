import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import time
import os

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class FaceEmotion(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.capture_interval = 1  # seconds
        self.last_capture_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Image in grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 0), thickness=2)

        current_time = time.time()
        elapsed_time = current_time - self.last_capture_time

        # Capture an image every specified interval
        if elapsed_time >= self.capture_interval:
            for i, (x, y, w, h) in enumerate(faces):
                face_image = img[y:y+h, x:x+w]
                image_path = f"images/capture_{int(current_time)}.jpg"
                cv2.imwrite(image_path, face_image)
                st.success(f"Image captured and saved: {image_path}")
            self.last_capture_time = current_time

        return img
    

def main():
    # Face Analysis Application #
    st.title("Real Time Face Detection Application Using IOT BOT")
    activities = ["Webcam Face Detection",'PI CAMERA DETECTION', 'Image Collection',"About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_transformer_factory=FaceEmotion)

                

    elif choice =='PI CAMERA DETECTION':
        
        pass
   
    elif choice == "Image Collection":

        
        image_dir = "images"
        image_files = os.listdir(image_dir)
        
        # Set the maximum number of images per row
        max_images_per_row = 5

        # Create columns for layout
        columns = st.columns(max_images_per_row)

        for i, image_file in enumerate(image_files):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, image_file)
                columns[i % max_images_per_row].image(image_path, width=150)
                
        # pass
    else:
        pass

if __name__ == "__main__":
    main()
