import cv2
import numpy as np
import os
import argparse
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import tempfile
import time


class YOLOv7:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        # Initialize model
        self.net = cv2.dnn.readNetFromONNX(path)
        # print("[INFO] setting preferable backend and target to CUDA...")
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        input_shape = os.path.splitext(os.path.basename(path))[0].split('_')[-1].split('x')
        self.input_height = int(input_shape[0])
        self.input_width = int(input_shape[1])
        
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.has_postprocess = 'score' in self.output_names
    
    def detect(self, image):
        input_img = self.prepare_input(image)
        blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)
        # Perform inference on the image
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outputs = self.net.forward(self.output_names)
        
        if self.has_postprocess:
            boxes, scores, class_ids = self.parse_processed_output(outputs)
        
        else:
            # Process output data
            boxes, scores, class_ids = self.process_output(outputs)
        
        return boxes, scores, class_ids
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        # Scale input pixel values to 0 to 1
        return input_img
    
    def process_output(self, output):
        predictions = np.squeeze(output[0])
        
        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]
        
        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]
        
        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)
        
        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]
        
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        a = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        if isinstance(a, tuple):
            return [],[],[]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold).flatten()
        
        return boxes[indices], scores[indices], class_ids[indices]
    
    def parse_processed_output(self, outputs):
        
        scores = np.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs[self.output_names.index('batchno_classid_x1y1x2y2')]
        
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]
        
        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]
        
        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]
        
        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        
        return boxes, scores, class_ids
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        
        # Convert boxes to xyxy format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        boxes_[..., 2] = boxes[..., 0] + boxes[..., 2] * 0.5
        boxes_[..., 3] = boxes[..., 1] + boxes[..., 3] * 0.5
        
        return boxes_
    
    def rescale_boxes(self, boxes):
        
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def draw_detections(self, image, boxes, scores, class_ids, distance):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'+ ' ' + f"{distance:.2f}m"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return image


 
    def estimate_distance(self, image, boxes, scores, class_ids, focal_length, known_width):
        distance = 0.0
        for box, conf, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            width = x2 - x1
            distance = (known_width * focal_length) / width
        return distance
    

'''
    def visualize_distance_multiframe(self, results, focal_length, known_width, num_frames=5):
        distances = []
        for *box, conf, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            width = x2 - x1
            distance = (known_width * focal_length) / width
            distances.append(distance)
     
            if len(distances) > num_frames:
                distances.pop(0)
     
            avg_distance = np.mean(distances)
'''



class SpeechInteraction:
    def __init__(self):
        # 初始化语音识别引擎
        # Initialize the speech recognition engine
        self.recEngine = sr.Recognizer()

    def recognize_audio(self):
        # 接收音频输入
        # Receive audio input
        with sr.Microphone() as source:
            print("Please start talking...")
            self.recEngine.adjust_for_ambient_noise(source, duration=1)  # 降低环境噪音影响
                                                                         # Reduced environmental noise impact

            try:
                #audio = self.recEngine.listen(source, timeout=1, phrase_time_limit=5)
                audio = self.recEngine.listen(source)
                text = self.recEngine.recognize_google(audio, language="en-EN")
                return text.lower()

                # 处理语音识别结果
                # Processing speech recognition results
                #self.recEngine_SpeechRecognized(text, distance)

            except sr.UnknownValueError:
                print("Unrecognized audio. Please try again.")
                return None

    def speak(self, text):
        # 使用 gTTS 进行语音合成
        # Speech synthesis using gTTS
        tts = gTTS(text=text, lang="en")

        # 保存临时音频文件
        # Saving temporary audio files
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)

        # 使用 pygame 实现实时播放语音
        # Real-time voice playback with pygame
        pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)

        # 删除临时音频文件
        # Delete temporary audio files
        os.remove(temp_filename)

    def prompt_distance_message(self, distance):
        if distance <= 2:
            self.speak("Obstructions present, please drive slowly")  # Obstacle ahead
        else:
            self.speak("No obstructions. Please proceed.")  # You can proceed
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/person.jpg', help="image path")
    # parser.add_argument('--modelpath', type=str, default='models/yolov7_256x640.onnx', help="onnx filepath")
    parser.add_argument('--modelpath', type=str, default='models/yolov7-tiny_640x640.onnx', help="onnx filepath")

    # parser.add_argument('--modelpath', type=str, default='models/yolov7_640x640.onnx',
    #                     choices=["models/yolov7_640x640.onnx", "models/yolov7-tiny_640x640.onnx",
    #                              "models/yolov7_736x1280.onnx", "models/yolov7-tiny_384x640.onnx",
    #                              "models/yolov7_480x640.onnx", "models/yolov7_384x640.onnx",
    #                              "models/yolov7-tiny_256x480.onnx", "models/yolov7-tiny_256x320.onnx",
    #                              "models/yolov7_256x320.onnx", "models/yolov7-tiny_256x640.onnx",
    #                              "models/yolov7_256x640.onnx", "models/yolov7-tiny_480x640.onnx",
    #                              "models/yolov7-tiny_736x1280.onnx", "models/yolov7_256x480.onnx"],
    #                     help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()
    
    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    speech_interaction = SpeechInteraction()
        
    vid = cv2.VideoCapture(0)
    focal_length = 700   # 焦距，需要根据实际情况进行调整
    known_width = 0.5   # 目标物体的实际宽度，根据实际情况进行调整

    while (True):
        ret, frame = vid.read()
        # Detect Objects
        boxes, scores, class_ids = yolov7_detector.detect(frame)
        distance = yolov7_detector.estimate_distance(frame, boxes, scores, class_ids, focal_length, known_width)


        if distance <= 2:
            speech_interaction.speak("Obstructions present, please drive slowly")  # Obstacle ahead
            while(distance<=1):
                ret, frame = vid.read()
                boxes, scores, class_ids = yolov7_detector.detect(frame)
                distance = yolov7_detector.estimate_distance(frame, boxes, scores, class_ids, focal_length, known_width)

                # Draw detections
                dstimg = yolov7_detector.draw_detections(frame, boxes, scores, class_ids, distance)
                winName = 'Deep learning object detection in OpenCV'
                # cv2.namedWindow(winName, 0)
                cv2.imshow(winName, dstimg)
                #print("end")
                cv2.waitKey(1)
                #print("over")
            #cv2.waitKey(1)
        
        if distance > 2:
            speech_interaction.speak("No obstructions. Please proceed.")  # You can proceed
            cv2.waitKey(1)
        #speech_interaction.prompt_distance_message(distance)

'''
        text = speech_interaction.recognize_audio()
        if text:
            # 判断是否识别到“关机”关键字，如果是则退出循环
            # Determine if the "Turn off" keyword is recognized, if so exit the loop.
            if text == "Turn off":
                break
            #elif distance <=2:
                #print("Obstacle ahead. Slow down.")
                #speech_interaction.speak("Obstacle ahead. Slow down.")
            elif text == "hello":
                print("Hello, nice to serve you.")
                speech_interaction.speak("Hello, nice to serve you.")
            elif text == "bye":
                print("Looking forward to serving you again. Goodbye.")
                speech_interaction.speak("Looking forward to serving you again. Goodbye.")        
'''

        
'''        
        # Draw detections
        dstimg = yolov7_detector.draw_detections(frame, boxes, scores, class_ids, distance)
        winName = 'Deep learning object detection in OpenCV'
        # cv2.namedWindow(winName, 0)
        cv2.imshow(winName, dstimg)
        cv2.waitKey(1)
'''
        
        #speech_interaction.recognize_audio(distance)

'''
    srcimg = cv2.imread(args.imgpath)
    
    # Detect Objects
    #results = yolov7_detector.detect(frame)
    boxes, scores, class_ids = yolov7_detector.detect(srcimg)
    
    # Draw detections
    dstimg = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)
    winName = 'Deep learning object detection in OpenCV'
    
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
