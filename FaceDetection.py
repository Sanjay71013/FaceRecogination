import cv2

"""
 This algorithm is only 90 percent accurate,
  Haar Cascade Algorithm:

 -> First get the image
 -> COnvert it into greyscale ( U can also print the grayscale image)
 -> get the face coordinates
 -> interate the face coordinates throuag a loop to draw rectangles around the face
 -> print the image
"""

# Load some pre-trained data frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in (image read function)
img = cv2.imread('img2.jpg')

# To print the image
# This line is to show the image but it shows the image for only milli second So it won't be visible for us
cv2.imshow('RDJ', img)
cv2.waitKey()  # This function pauses the above line until we press a key so that we will be able to see the image

# Must convert to grayscale (convert color function)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('RDJ',grayscaled_img)
# cv2.waitKey()

""" In this cv2 module RGB is reverse. So always fill like (B,G,R)"""

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles

# print(face_coordinates) --- prints  [[253  92 262 262]] --[[x y w h]]

# Draw rectangles around the faces

# cv2.rectangle(img, (253, 92), (253+262, 92+262), (0, 255, 0), 2) --- (img, (x,y) ,(x+w, y+h), color_of_rectangle, thickness_of_the_line)
'''''
(x, y, w, h) = face_coordinates[1]  # It's a nested list
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
'''''
# The above 2 lines can detect only 1 face in the picture. If we want to find all the faces we should use loop

# For multiple faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow('FaceDetection', img)
cv2.waitKey()

print("code completed!")
