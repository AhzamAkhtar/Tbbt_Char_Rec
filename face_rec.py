import cv2 as cv
haar_cascade=cv.CascadeClassifier("haar_face.xml")
people=["lenord","sheldon"]
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_tbbt.yml")
img=cv.imread(r"C:\Users\ahzam\PycharmProjects\tbbt\val\sheldon\the-big-bang-theory-final-sheldon-cooper-jim-parsons-1561048451.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("person",gray)
face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for (x,y,w,h) in face_rect:
    faces_roi=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f"label={people[label]} with a confidence of {confidence}")

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("Dect",img)
cv.waitKey(0)


