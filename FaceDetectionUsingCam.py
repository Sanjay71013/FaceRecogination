import cv2

"""
 This algorithm is only 90 percent accurate
 Cascade Algorithm
"""

# Load some pre-trained data frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


""" 
# Choose an image to detect faces in (image read function)
img = cv2.imread('img2.jpg')
"""

# To capture video from webcam
# 0 is to indcate to go to default web cam
""" Video= cv2.VideoCapture('(location)video.mp4') --- To capture faces from a video"""
webcam = cv2.VideoCapture(0)


# TO loop through all the frames in the video
while True:

    # Read the current frame
    # successful_frame_read is a boolean value
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale (convert color function)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    # For multiple faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('FaceDetectionUsingCam', frame)
    # 1 indicates that each frame will be printed for 1 millisecond, otherwise we have to press a key for each frame
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

print("code completed!")
