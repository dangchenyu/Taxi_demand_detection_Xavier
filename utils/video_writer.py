import cv2
def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 14, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()