import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from datetime import datetime
import yaml
import re

class ImageCapture(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        self.bridge = CvBridge()

        self.current_frame = None
        self.current_frame_copy = None
        self.display_frame = None
        self.photo_count = 0
        self.current_class = "object"
        self.roi_points = []
        self.drawing = False
        self.class_id_map = {}
        self.next_class_id = 0
        self.in_roi_mode = False
        
        self.dataset_number = None
        self.dataset_dir = None

        # Выбираем или создаем датасет
        self.choose_or_create_dataset()
        
        self.photo_dir = self.dataset_dir
        self.images_dir = os.path.join(self.photo_dir, 'images')
        self.labels_dir = os.path.join(self.photo_dir, 'labels')
        
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)
        
        self.get_logger().info(f'Created directories in: {self.photo_dir}')

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_color',
            self.image_callback,
            10
        )
        self.get_logger().info("Подписались на топик с изображением")
        self.print_help()
    
    def print_help(self):
        help_text = """
╔════════════════════════════════════════════════╗
║          Управление захватом изображений       ║
╠════════════════════════════════════════════════╣
║ SPACE       - Начать выделение ROI             ║
║ ESC         - Отменить выделение               ║
║ ENTER       - Сохранить фото с ROI             ║
║ C           - Изменить класс объекта           ║
║ Q           - Выход из программы               ║
║                                                ║
║ Текущий класс: {}                          ║
║ Датасет: dataset_{}                             ║
╚════════════════════════════════════════════════╝
        """.format(self.current_class, self.dataset_number)
        self.get_logger().info(help_text)
    
    def find_next_dataset_number(self):
        """Находит следующий номер датасета"""
        cwd = os.getcwd()
        existing_datasets = []
        
        for folder in os.listdir(cwd):
            match = re.match(r'dataset_(\d+)', folder)
            if match:
                existing_datasets.append(int(match.group(1)))
        
        if not existing_datasets:
            return 1
        return max(existing_datasets) + 1
    
    def load_existing_dataset(self, dataset_path):
        """Загружает существующий датасет и восстанавливает состояние"""
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        
        # Загружаем классы из yaml
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'names' in config:
                    # Восстанавливаем маппинг классов
                    self.class_id_map = {v: k for k, v in config['names'].items()}
                    self.next_class_id = len(self.class_id_map)
        
        # Подсчитываем существующие фото
        images_dir = os.path.join(dataset_path, 'images')
        if os.path.exists(images_dir):
            self.photo_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        
        self.get_logger().info(f'Загружен датасет: {dataset_path}')
        self.get_logger().info(f'Классы: {self.class_id_map}')
        self.get_logger().info(f'Существующих фото: {self.photo_count}')
    
    def choose_or_create_dataset(self):
        """Спрашивает пользователя: дополнить датасет или создать новый"""
        cwd = os.getcwd()
        existing_datasets = []
        
        # Ищем существующие датасеты
        for folder in sorted(os.listdir(cwd)):
            if folder.startswith('dataset_') and os.path.isdir(os.path.join(cwd, folder)):
                match = re.match(r'dataset_(\d+)', folder)
                if match:
                    existing_datasets.append((int(match.group(1)), folder))
        
        existing_datasets.sort()
        
        print("\n" + "="*60)
        if existing_datasets:
            print("Найдены существующие датасеты:")
            for idx, (num, folder_name) in enumerate(existing_datasets):
                dataset_path = os.path.join(cwd, folder_name)
                images_count = len([f for f in os.listdir(os.path.join(dataset_path, 'images')) 
                                   if os.path.exists(os.path.join(dataset_path, 'images')) 
                                   and f.endswith(('.jpg', '.png'))])
                print(f"  {idx + 1}. {folder_name} (фото: {images_count})")
            
            choice = input("\nВыберите датасет (номер) или нажмите Enter для создания нового: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(existing_datasets):
                selected_dataset = existing_datasets[int(choice) - 1]
                self.dataset_number = selected_dataset[0]
                self.dataset_dir = os.path.join(cwd, selected_dataset[1])
                self.load_existing_dataset(self.dataset_dir)
            else:
                # Создаем новый датасет
                self.dataset_number = self.find_next_dataset_number()
                self.dataset_dir = os.path.join(cwd, f'dataset_{self.dataset_number}')
                os.makedirs(self.dataset_dir, exist_ok=True)
                self.get_logger().info(f'Создан новый датасет: dataset_{self.dataset_number}')
        else:
            # Нет существующих датасетов, создаем первый
            self.dataset_number = 1
            self.dataset_dir = os.path.join(cwd, f'dataset_{self.dataset_number}')
            os.makedirs(self.dataset_dir, exist_ok=True)
            self.get_logger().info(f'Создан новый датасет: dataset_{self.dataset_number}')
        
        print("="*60 + "\n")
    
    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame_copy = self.current_frame.copy()
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points = [(x, y)]
            self.drawing = True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current_frame_copy is not None and len(self.roi_points) > 0:
                self.display_frame = self.current_frame_copy.copy()
                cv2.rectangle(self.display_frame, self.roi_points[0], (x, y), (0, 255, 0), 2)
                cv2.putText(self.display_frame, f'Class: {self.current_class}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_points.append((x, y))
            self.drawing = False
    
    def change_class(self):
        print("\n" + "="*50)
        print(f"Текущий класс: {self.current_class}")
        new_class = input("Введите новое имя класса (или нажмите Enter для отмены): ").strip()
        if new_class:
            self.current_class = new_class
            if new_class not in self.class_id_map:
                self.class_id_map[new_class] = self.next_class_id
                self.next_class_id += 1
            self.get_logger().info(f"Класс изменен на: {self.current_class}")
        print("="*50 + "\n")
    
    def save_photo_with_roi(self):
        if self.current_frame is None or len(self.roi_points) < 2:
            self.get_logger().warn("ROI не выделена или изображение не получено")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'photo_{self.photo_count:03d}_{timestamp}'
        image_path = os.path.join(self.images_dir, filename + '.jpg')
        label_path = os.path.join(self.labels_dir, filename + '.txt')
        
        # Сохраняем фото
        cv2.imwrite(image_path, self.current_frame)
        
        # Нормализуем координаты
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        img_width = self.current_frame.shape[1]
        img_height = self.current_frame.shape[0]
        
        # YOLO формат: <class_id> <x_center> <y_center> <width> <height>
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Добавляем класс в маппинг если его еще нет
        if self.current_class not in self.class_id_map:
            self.class_id_map[self.current_class] = self.next_class_id
            self.next_class_id += 1
        
        class_id = self.class_id_map[self.current_class]
        
        # Сохраняем label в YOLO формате
        with open(label_path, 'w') as f:
            f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
        
        self.photo_count += 1
        self.roi_points = []
        self.get_logger().info(f'Сохранено: {image_path}')
        self.get_logger().info(f'Label: {label_path}')
    
    def save_dataset_yaml(self):
        """Сохраняет dataset.yaml для YOLO"""
        if self.photo_count == 0:
            self.get_logger().warn("Фото не были сохранены, конфиг не создан")
            return
        
        yaml_path = os.path.join(self.photo_dir, 'dataset.yaml')
        
        dataset_config = {
            'path': self.photo_dir,
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': len(self.class_id_map),
            'names': {v: k for k, v in self.class_id_map.items()}
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        self.get_logger().info(f'Сохранен конфиг: {yaml_path}')

def main(args=None):
    rclpy.init(args=args)
    image_capture = ImageCapture()

    window_name = 'Camera Feed - Press SPACE to draw ROI, C to change class, Q to exit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, image_capture.mouse_callback)
    
    roi_window_name = 'Draw ROI - ESC to cancel, ENTER to save'

    try:
        while True:
            rclpy.spin_once(image_capture, timeout_sec=0.01)

            if image_capture.current_frame is not None:
                frame_display = image_capture.current_frame.copy()
                
                cv2.putText(frame_display, f'Class: {image_capture.current_class}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_display, f'Photos: {image_capture.photo_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if len(image_capture.roi_points) == 2:
                    cv2.rectangle(frame_display, image_capture.roi_points[0], 
                                image_capture.roi_points[1], (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Q key
                    image_capture.get_logger().info("Выход из программы")
                    break
                elif key == ord('c'):  # C key
                    image_capture.change_class()
                elif key == 32:  # SPACE key
                    image_capture.roi_points = []
                    image_capture.in_roi_mode = True
                    cv2.namedWindow(roi_window_name)
                    cv2.setMouseCallback(roi_window_name, image_capture.mouse_callback)
                    
                    while image_capture.in_roi_mode:
                        rclpy.spin_once(image_capture, timeout_sec=0.01)
                        
                        if image_capture.display_frame is not None:
                            cv2.imshow(roi_window_name, image_capture.display_frame)
                        elif image_capture.current_frame_copy is not None:
                            frame_temp = image_capture.current_frame_copy.copy()
                            cv2.putText(frame_temp, f'Class: {image_capture.current_class}', 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(roi_window_name, frame_temp)
                        
                        key_roi = cv2.waitKey(1) & 0xFF
                        if key_roi == 27:  # ESC
                            image_capture.roi_points = []
                            image_capture.display_frame = None
                            image_capture.in_roi_mode = False
                            cv2.destroyWindow(roi_window_name)
                            break
                        elif key_roi == 13:  # ENTER
                            image_capture.save_photo_with_roi()
                            image_capture.display_frame = None
                            image_capture.in_roi_mode = False
                            cv2.destroyWindow(roi_window_name)
                            break
    
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        image_capture.save_dataset_yaml()
        image_capture.get_logger().info(f'Всего сохранено фото: {image_capture.photo_count}')
        image_capture.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()