import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode


# # Load the pre-trained face detector from OpenCV
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to detect faces in the image
# def detect_faces(image):
#     grayscaled_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
#     return face_coordinates

# # Video processor class for face detection
# class FaceDetectionProcessor(VideoProcessorBase):
#     def __init__(self):
#         super().__init__()

#     def recv(self, frame: np.ndarray, frame_metadata, sender_id: str):
#         # Detect faces
#         face_coordinates = detect_faces(frame)

#         # Draw rectangles around the detected faces
#         for (x, y, w, h) in face_coordinates:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         return frame

# # Streamlit app
# def main():
#     st.title("Face Detection with Webcam using streamlit_webrtc")

#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTCConfiguration(
#             iceServers=[
#                 {"urls": ["stun:stun.l.google.com:19302"]},
#             ],
#         ),
#         video_processor_factory=FaceDetectionProcessor,
#         async_processing=True,
#     )

#     if webrtc_ctx.video_processor:
#         st.image(webrtc_ctx.video_processor_factory.frame, channels="BGR", use_column_width=True, caption="Webcam Face Detection")

# # Run the app
# if __name__ == "__main__":
#     main()



import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Load the pre-trained face detector from OpenCV
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video transformer class for face detection
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def transform(self, frame: np.ndarray) -> np.ndarray:
        # Detect faces
        face_coordinates = self.detect_faces(frame)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame

    def detect_faces(self, image):
        grayscaled_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
        return face_coordinates

# Streamlit app
def main():
    st.title("Face Detection with Webcam using streamlit_webrtc")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            iceServers=[
                {"urls": ["stun:stun.l.google.com:19302"]},
            ],
        ),
        video_transformer_factory=FaceDetectionTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        st.image(webrtc_ctx.video_transformer.frame_out, channels="BGR", use_column_width=True, caption="Webcam Face Detection")

# Run the app
if __name__ == "__main__":
    main()
