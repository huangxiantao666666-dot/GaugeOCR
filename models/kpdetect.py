"""
YOLO26-pose 人体关键点检测 Demo
使用 COCO 预训练模型，检测 17 个人体关键点
支持：图片、视频、摄像头
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


# ==================== 配置 ====================
# COCO 人体关键点名称（17个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点连接关系（骨架）
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),               # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)    # 腿部
]

# 颜色配置
COLOR_KEYPOINT = (0, 255, 0)      # 关键点：绿色
COLOR_SKELETON = (255, 128, 0)    # 骨架：橙色
COLOR_BOX = (255, 0, 0)           # 检测框：蓝色


# ==================== 工具函数 ====================
def draw_pose(image, results, conf_threshold=0.5):
    """
    在图像上绘制人体关键点和骨架
    """
    for result in results:
        if result.keypoints is None:
            continue
            
        # 获取关键点数据 [N, 17, 3] -> x, y, conf
        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None
        
        for i, kpt in enumerate(keypoints):
            # 绘制检测框
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_BOX, 2)
                cv2.putText(image, f"person {result.boxes.conf[i].item():.2f}", 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 2)
            
            # 绘制骨架（连线）
            valid_points = []
            for idx, (x, y, conf) in enumerate(kpt):
                if conf > conf_threshold:
                    valid_points.append((idx, (int(x), int(y))))
            
            for start_idx, end_idx in SKELETON:
                start_point = None
                end_point = None
                for idx, pt in valid_points:
                    if idx == start_idx:
                        start_point = pt
                    if idx == end_idx:
                        end_point = pt
                if start_point is not None and end_point is not None:
                    cv2.line(image, start_point, end_point, COLOR_SKELETON, 2)
            
            # 绘制关键点
            for idx, (x, y, conf) in enumerate(kpt):
                if conf > conf_threshold:
                    cv2.circle(image, (int(x), int(y)), 4, COLOR_KEYPOINT, -1)
    
    return image


def process_image(model, image_path, conf_threshold=0.5, save_path=None):
    """
    处理单张图片
    """
    results = model(image_path, conf=conf_threshold)
    image = cv2.imread(str(image_path))
    image = draw_pose(image, results, conf_threshold)
    
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"结果已保存至: {save_path}")
    
    cv2.imshow("YOLO26-pose Demo", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 打印关键点信息
    for result in results:
        if result.keypoints is not None:
            print(f"检测到 {len(result.keypoints)} 个人体")
            print(f"关键点数据形状: {result.keypoints.data.shape}")
            print(f"关键点坐标: {result.keypoints.xy}")
    return results


def process_video(model, source, conf_threshold=0.5):
    """
    处理视频或摄像头
    source: 视频文件路径 或 摄像头ID（如0）
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开视频源: {source}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置输出视频保存
    output_path = "pose_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"开始处理视频，按 'q' 退出，结果保存至 {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame, conf=conf_threshold)
        frame = draw_pose(frame, results, conf_threshold)
        
        # 显示
        cv2.imshow("YOLO26-pose Demo", frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("处理完成")


def main():
    parser = argparse.ArgumentParser(description="YOLO26-pose 人体关键点检测 Demo")
    parser.add_argument("--source", type=str, required=True,
                       help="输入源：图片路径、视频路径 或 摄像头ID(0)")
    parser.add_argument("--model", type=str, default="../checkpoints/yolo26m-pose.pt",
                       help="模型路径，可选: yolo26n-pose.pt, yolo26s-pose.pt, yolo26m-pose.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--save", type=str, default=None,
                       help="图片结果保存路径（仅图片模式有效）")
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 判断输入类型
    source_path = Path(args.source)
    if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(model, args.source, args.conf, args.save)
    else:
        # 尝试转为整数判断是否为摄像头
        try:
            camera_id = int(args.source)
            process_video(model, camera_id, args.conf)
        except ValueError:
            process_video(model, args.source, args.conf)


if __name__ == "__main__":
    main()