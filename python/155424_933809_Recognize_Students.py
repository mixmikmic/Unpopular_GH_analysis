import cv2
import os

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME1.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME1/'
video_to_frames(video_path, output_image_path)

# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME2.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME2/'
video_to_frames(video_path, output_image_path)

# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME3.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME3/'
video_to_frames(video_path, output_image_path)



