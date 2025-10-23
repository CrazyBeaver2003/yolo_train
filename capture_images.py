import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from datetime import datetime

class ImageCapture(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        self.bridge = CvBridge()

        self.current_frame = None
        self.photo_count = 0

        self.photo_dir = os.path.join(os.getcwd(), 'photos')
        if not os.path.exists(self.photo_dir):
            os.makedirs(self.photo_dir)
            self.get_logger().info(f'Created directory: {self.photo_dir}')

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_color',
            self.image_callback,
            10
        )
        self.get_logger().info("Подписались на топик c изображением")
        self.get_logger().info("Нажмите SPACE для сохранения фото, ESC для выхода")
    
    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
    
    def save_photo(self):
        if self.current_frame is None:
            self.get_logger().warn("Изображение еще не получено")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'photo_{self.photo_count:03d}_{timestamp}.jpg'
        filepath = os.path.join(self.photo_dir, filename)

        cv2.imwrite(filepath, self.current_frame)
        self.photo_count += 1
        self.get_logger().info(f'Сохранено фото: {filepath}')

def main(args=None):
    rclpy.init(args=args)
    image_capture = ImageCapture()

    try:
        while True:
            rclpy.spin_once(image_capture, timeout_sec=0.1)

            if image_capture.current_frame is not None:
                cv2.imshow('Camera Feed - Press SPACE to capture, ESC to exit', 
                        image_capture.current_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == 32:  # SPACE key
                    image_capture.save_photo()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        image_capture.get_logger().info(f'Photos saved: {image_capture.photo_count}')
        image_capture.destroy_node()

if __name__ == '__main__':
    main()