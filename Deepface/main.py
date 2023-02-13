from deepface import DeepFace


def faceVerification():
    result1 = DeepFace.verify(img1_path = "images/Biden1.jpeg", img2_path = "images/Biden2.jpeg")
    result2 = DeepFace.verify(img1_path = "images/Trump1.jpeg", img2_path = "images/Trump2.jpeg")
    result3 = DeepFace.verify(img1_path = "images/Biden1.jpeg", img2_path = "images/Trump1.jpeg")
    result4 = DeepFace.verify(img1_path = "images/Biden2.jpeg", img2_path = "images/Trump2.jpeg")
    result5 = DeepFace.verify(img1_path = "images/Mars1.jpeg", img2_path = "images/Mars2.jpeg")
    result6 = DeepFace.verify(img1_path = "images/Mars1.jpeg", img2_path = "images/Biden1.jpeg")
    result7 = DeepFace.verify(img1_path = "images/Mars1.jpeg", img2_path = "images/Trump1.jpeg")
    print("Biden1 and Biden2: " + str(result1))
    print("Trump1 and Trump2: " + str(result2))
    print("Biden1 and Trump1: " + str(result3))
    print("Biden2 and Trump2: " + str(result4))
    print("Mars1 and Mars1: " + str(result5))
    print("Mars1 and Biden1: " + str(result6))
    print("Mars1 and Trump1: " + str(result7))

def facialAttributeAnalysis():
    Biden = DeepFace.analyze(img_path = "images/Biden1.jpeg", actions = ['age', 'gender', 'race', 'emotion'])
    Trump = DeepFace.analyze(img_path = "images/Trump2.jpeg", actions = ['age', 'gender', 'race', 'emotion'])
    Mars = DeepFace.analyze(img_path = "images/Mars2.jpeg", actions = ['age', 'gender', 'race', 'emotion'])
    print("Biden: " + "age: " + str(Biden[0]['age']) + ", gender: " + str(Biden[0]['dominant_gender']) + ", race: " + str(Biden[0]['dominant_race']) + ", emotion: " + str(Biden[0]['dominant_emotion']))
    print("Trump: " + "age: " + str(Trump[0]['age']) + ", gender: " + str(Trump[0]['dominant_gender']) + ", race: " + str(Trump[0]['dominant_race']) + ", emotion: " + str(Trump[0]['dominant_emotion']))
    print("Mars: " + "age: " + str(Mars[0]['age']) + ", gender: " + str(Mars[0]['dominant_gender']) + ", race: " + str(Mars[0]['dominant_race']) + ", emotion: " + str(Mars[0]['dominant_emotion']))

def faceDetection():
    Biden1 = DeepFace.extract_faces(img_path = "images/Biden1.jpeg")
    Trump1 = DeepFace.extract_faces(img_path = "images/Trump1.jpeg")
    Mars1 = DeepFace.extract_faces(img_path = "images/Mars1.jpeg")
    print("Biden1: " + "facial area: " + str(Biden1[0]['facial_area']) + ", confidence: " + str(Biden1[0]['confidence']))
    print("Trump1: " + "facial area: " + str(Trump1[0]['facial_area']) + ", confidence: " + str(Trump1[0]['confidence']))
    print("Mars1: " + "facial area: " + str(Mars1[0]['facial_area']) + ", confidence: " + str(Mars1[0]['confidence']))


faceVerification()
facialAttributeAnalysis()
faceDetection()