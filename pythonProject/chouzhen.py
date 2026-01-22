import cv2
import os


def extract_frames_opencv(video_path, output_dir, interval=1):
    """
    使用OpenCV抽取视频帧
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval: 抽帧间隔（每秒多少帧）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"视频信息: {total_frames}帧, {fps:.2f}fps, {duration:.2f}秒")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 按间隔保存帧
        if frame_count % int(fps * interval) == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"已保存第 {saved_count} 帧")

        frame_count += 1

    cap.release()
    print(f"抽帧完成！共保存 {saved_count} 张图片")


# 使用示例
if __name__ == "__main__":
    extract_frames_opencv("E:/015_real.mp4", "E:/015", interval=1)  # 每秒1帧