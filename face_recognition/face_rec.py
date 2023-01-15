import numpy as np
import cv2
import face_recognition

vidio_capture = cv2.VideoCapture(0)

rakhmiddin_image = face_recognition.load_image_file("Ruzaliev.jpg")
rakhmiddin_face_encoding = face_recognition.face_encodings(rakhmiddin_image)[0]

obama_image = face_recognition.load_image_file("obama.jpg")
obama_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [
   
    rakhmiddin_face_encoding,
    obama_encoding,
    biden_encoding
]
known_face_names = [
    "Rakhmiddin",
    "Obama",
    "Biden"
]

while True:
    ret, frame = vidio_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches =face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unkown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top-50), (right, bottom), color=(0,0,255), thickness=2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom-6), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255,255,255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidio_capture.release()
cv2.destroyAllWindows()
print("Complete")