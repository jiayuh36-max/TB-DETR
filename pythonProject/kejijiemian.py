import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout,
    QVBoxLayout, QGridLayout, QTextEdit, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QLinearGradient, QBrush
from PyQt5.QtCore import Qt


class FancyLogin(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("登录")
        self.resize(900, 500)

        # 设置科技感背景
        self.setAutoFillBackground(True)
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(10, 20, 35))
        gradient.setColorAt(1, QColor(5, 10, 20))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # ===== 左右分栏 =====
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        right_panel = QLabel()
        right_panel.setAlignment(Qt.AlignCenter)
        right_panel.setObjectName("rightPanel")
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        # ===== 左侧内容 =====
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(80, 60, 80, 60)

        title = QLabel("不知道什么队")
        title.setObjectName("title")
        left_layout.addWidget(title, alignment=Qt.AlignHCenter)

        grid = QGridLayout()
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("账号")
        self.user_edit.setObjectName("input")

        self.pwd_edit = QLineEdit()
        self.pwd_edit.setPlaceholderText("密码")
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setObjectName("input")

        grid.addWidget(self.user_edit, 0, 0)
        grid.addWidget(self.pwd_edit, 1, 0)
        left_layout.addLayout(grid)

        self.login_btn = QPushButton("登录")
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self.check_login)
        left_layout.addWidget(self.login_btn, alignment=Qt.AlignHCenter)

        hint = QLabel("—— 人脸伪造识别系统 ——")
        hint.setObjectName("hint")
        left_layout.addWidget(hint, alignment=Qt.AlignHCenter)

        left_layout.addStretch()

        self.setStyleSheet(self.qss())

    def qss(self):
        return """
        #leftPanel {
            background-color: rgba(15, 25, 40, 180);
            border-radius: 10px;
            border: 1px solid rgba(0, 200, 255, 80);
        }
        #rightPanel {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                             stop:0 rgba(10, 30, 50, 255), 
                                             stop:1 rgba(5, 15, 25, 255));
            border-left: 1px solid rgba(0, 150, 255, 50);
        }
        #title {
            font-size: 24px;
            color: #00ccff;
            margin-bottom: 30px;
            font-weight: bold;
            letter-spacing: 2px;
        }
        #input {
            height: 40px;
            border: 1px solid #00aaff;
            border-radius: 4px;
            padding-left: 15px;
            margin-top: 15px;
            background: rgba(10, 25, 40, 200);
            color: #00ccff;
            font-size: 14px;
        }
        #input:focus {
            border: 1px solid #00ffff;
            background: rgba(15, 35, 55, 220);
        }
        #login_btn {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 150, 255, 255), 
                                             stop:1 rgba(0, 100, 200, 255));
            color: #ffffff;
            border-radius: 4px;
            height: 40px;
            width: 180px;
            margin-top: 30px;
            border: none;
            font-weight: bold;
            letter-spacing: 1px;
        }
        #login_btn:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 180, 255, 255), 
                                             stop:1 rgba(0, 120, 220, 255));
        }
        #login_btn:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 120, 220, 255), 
                                             stop:1 rgba(0, 80, 180, 255));
        }
        #hint {
            margin-top: 30px;
            color: #5599bb;
            font-size: 12px;
            letter-spacing: 1px;
        }
        """

    def check_login(self):
        user = self.user_edit.text().strip()
        pwd = self.pwd_edit.text()
        if user == "1" and pwd == "1":
            self.main_window = MainWindow()
            self.main_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "认证失败", "用户名或密码错误")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("多媒体处理系统")
        self.resize(1200, 800)

        # 设置科技感背景
        self.setAutoFillBackground(True)
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(10, 20, 35))
        gradient.setColorAt(1, QColor(5, 10, 20))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 左侧面板 - 文件选择
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)

        # 使用选项卡分隔图片和视频
        tab_widget = QTabWidget()
        tab_widget.setObjectName("tabWidget")

        # 图片选项卡
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)

        image_title = QLabel("图片文件")
        image_title.setObjectName("tabTitle")
        image_layout.addWidget(image_title)

        self.image_btn = QPushButton("添加图片")
        self.image_btn.setObjectName("addBtn")
        self.image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.image_btn)

        self.image_list = QListWidget()
        self.image_list.setObjectName("fileList")
        self.image_list.itemClicked.connect(lambda item: self.display_selected_file(item, 'image'))
        image_layout.addWidget(self.image_list)

        # 视频选项卡
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)

        video_title = QLabel("视频文件")
        video_title.setObjectName("tabTitle")
        video_layout.addWidget(video_title)

        self.video_btn = QPushButton("添加视频")
        self.video_btn.setObjectName("addBtn")
        self.video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.video_btn)

        self.video_list = QListWidget()
        self.video_list.setObjectName("fileList")
        self.video_list.itemClicked.connect(lambda item: self.display_selected_file(item, 'video'))
        video_layout.addWidget(self.video_list)

        # 添加选项卡
        tab_widget.addTab(image_tab, "🖼️ 图片")
        tab_widget.addTab(video_tab, "📹 视频")

        left_layout.addWidget(tab_widget)

        # 中间面板 - 两个显示区域
        center_panel = QWidget()
        center_panel.setObjectName("centerPanel")
        center_layout = QVBoxLayout(center_panel)

        # 上面显示区域 - 原始样式
        top_display_title = QLabel("原始内容显示")
        top_display_title.setObjectName("panelTitle")
        center_layout.addWidget(top_display_title)

        self.original_display = QLabel()
        self.original_display.setObjectName("originalDisplay")
        self.original_display.setAlignment(Qt.AlignCenter)
        self.original_display.setText("原始文件将显示在这里")
        self.original_display.setMinimumHeight(200)
        center_layout.addWidget(self.original_display)

        # 下面显示区域 - 可点击选择文件
        bottom_display_title = QLabel("处理后结果显示 ")
        bottom_display_title.setObjectName("panelTitle")
        center_layout.addWidget(bottom_display_title)

        # 创建一个容器来包装可点击的标签
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)

        self.result_display = QLabel()
        self.result_display.setObjectName("resultDisplay")
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setText("处理结果将显示在这里")
        self.result_display.setMinimumHeight(200)
        self.result_display.mousePressEvent = self.result_display_clicked
        result_layout.addWidget(self.result_display)

        center_layout.addWidget(result_container)

        # 状态信息显示
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.status_label)

        # 开始处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.setObjectName("processBtn")
        self.process_btn.clicked.connect(self.process_file)
        center_layout.addWidget(self.process_btn)

        # 右侧面板 - 只保留文本框
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)

        # 右侧标题
        right_title = QLabel("伪造区域提示")
        right_title.setObjectName("panelTitle")
        right_layout.addWidget(right_title)

        # 文本输入框
        self.text_input = QTextEdit()
        self.text_input.setObjectName("textInput")
        right_layout.addWidget(self.text_input)

        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel)
        main_layout.addWidget(right_panel)

        self.setStyleSheet(self.qss())

        # 存储当前选择的文件
        self.current_file = None
        self.current_file_type = None
        # 存储要处理的文件（通过点击下面区域选择的）
        self.selected_process_file = None
        self.selected_process_file_type = None

    def qss(self):
        return """
        QWidget {
            font-family: "Segoe UI";
            color: #aaddff;
        }
        #leftPanel, #centerPanel, #rightPanel {
            background-color: rgba(15, 25, 40, 200);
            border-radius: 5px;
            border: 1px solid rgba(0, 150, 255, 50);
            padding: 15px;
        }
        #panelTitle {
            font-size: 16px;
            color: #00ccff;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(0, 150, 255, 80);
        }
        #tabTitle {
            font-size: 14px;
            color: #00ccff;
            font-weight: bold;
            margin-bottom: 8px;
        }
        #addBtn {
            background-color: rgba(0, 120, 220, 180);
            color: white;
            border-radius: 4px;
            height: 28px;
            padding: 0 12px;
            margin-bottom: 8px;
            border: 1px solid rgba(0, 150, 255, 100);
        }
        #addBtn:hover {
            background-color: rgba(0, 150, 255, 220);
        }
        #fileList {
            background-color: rgba(10, 20, 35, 200);
            border: 1px solid rgba(0, 150, 255, 80);
            border-radius: 4px;
            padding: 5px;
            color: #aaddff;
        }
        #fileList::item {
            padding: 6px;
            border-bottom: 1px solid rgba(0, 100, 200, 50);
            font-size: 11px;
        }
        #fileList::item:selected {
            background-color: rgba(0, 120, 220, 150);
            color: #ffffff;
        }
        #tabWidget::pane {
            border: 1px solid rgba(0, 150, 255, 80);
            border-radius: 4px;
            background: rgba(10, 20, 35, 200);
        }
        #tabWidget QTabBar::tab {
            background: rgba(0, 80, 160, 150);
            color: #aaddff;
            padding: 6px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
            font-size: 12px;
            border: 1px solid rgba(0, 150, 255, 80);
        }
        #tabWidget QTabBar::tab:selected {
            background: rgba(0, 120, 220, 200);
            color: #ffffff;
        }
        #originalDisplay {
            background-color: rgba(10, 20, 35, 150);
            border: 1px dashed rgba(0, 150, 255, 100);
            border-radius: 4px;
            padding: 10px;
            color: #88bbdd;
            font-size: 13px;
        }
        #resultDisplay {
            background-color: rgba(10, 20, 35, 150);
            border: 2px dashed rgba(0, 200, 255, 100);
            border-radius: 4px;
            padding: 20px;
            color: #88bbdd;
            font-size: 14px;
        }
        #textInput {
            background-color: rgba(10, 20, 35, 200);
            border: 1px solid rgba(0, 150, 255, 100);
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
            color: #aaddff;
        }
        #statusLabel {
            background-color: rgba(10, 20, 35, 150);
            border: 1px solid rgba(0, 150, 255, 80);
            border-radius: 4px;
            padding: 6px;
            margin: 8px 0;
            font-size: 11px;
            color: #88bbdd;
        }
        #processBtn {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 150, 255, 255), 
                                             stop:1 rgba(0, 100, 200, 255));
            color: #ffffff;
            border-radius: 4px;
            height: 35px;
            font-size: 14px;
            margin-top: 10px;
            margin-bottom: 10px;
            border: none;
            font-weight: bold;
        }
        #processBtn:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 180, 255, 255), 
                                             stop:1 rgba(0, 120, 220, 255));
        }
        #processBtn:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, 
                                             stop:0 rgba(0, 120, 220, 255), 
                                             stop:1 rgba(0, 80, 180, 255));
        }
        """

    def result_display_clicked(self, event):
        """处理结果显示区域的点击事件"""
        if event.button() == Qt.LeftButton:
            self.select_process_file()

    def select_process_file(self):
        """通过点击下面显示区域选择要处理的文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择要处理的文件", "",
                "媒体文件 (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.selected_process_file = file_path

                # 判断文件类型
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.selected_process_file_type = 'image'
                else:
                    self.selected_process_file_type = 'video'

        except Exception as e:
            print(f"选择文件错误: {e}")
            QMessageBox.warning(self, "错误", "选择文件时发生错误")

    def select_video(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                item = QListWidgetItem(f"📹 {file_path.split('/')[-1]}")
                item.setData(Qt.UserRole, file_path)
                self.video_list.addItem(item)
        except Exception as e:
            print(f"选择视频错误: {e}")

    def select_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            if file_path:
                item = QListWidgetItem(f"🖼️ {file_path.split('/')[-1]}")
                item.setData(Qt.UserRole, file_path)
                self.image_list.addItem(item)
        except Exception as e:
            print(f"选择图片错误: {e}")

    def display_selected_file(self, item, file_type):
        """从左侧列表选择文件显示在上面区域"""
        try:
            file_path = item.data(Qt.UserRole)
            self.current_file = file_path

            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.current_file_type = 'image'
                pixmap = QPixmap(file_path)

                # 显示在上面原始区域
                display_width = self.original_display.width() - 40
                display_height = self.original_display.height() - 40
                if display_width > 0 and display_height > 0:
                    scaled_pixmap = pixmap.scaled(
                        display_width,
                        display_height,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.original_display.setPixmap(scaled_pixmap)
                else:
                    self.original_display.setPixmap(pixmap)
                self.original_display.setText("")
                self.status_label.setText(f"📷 原始图片: {file_path.split('/')[-1]}")

            else:
                self.current_file_type = 'video'
                # 显示在上面原始区域
                self.original_display.clear()
                self.original_display.setPixmap(QPixmap())
                self.original_display.setText("📹 视频文件\n\n原始视频内容")
                self.status_label.setText(f"🎥 原始视频: {file_path.split('/')[-1]}")

        except Exception as e:
            print(f"显示文件错误: {e}")

    def process_file(self):
        """开始处理文件"""
        if not self.selected_process_file:
            QMessageBox.warning(self, "提示", "请先点击下面区域选择一个要处理的文件")
            return

        try:
            self.status_label.setText("处理中...")

            # 模拟处理过程
            if self.selected_process_file_type == 'image':
                # 显示选择的图片（不是原始图片）
                pixmap = QPixmap(self.selected_process_file)
                display_width = self.result_display.width() - 40
                display_height = self.result_display.height() - 40
                if display_width > 0 and display_height > 0:
                    scaled_pixmap = pixmap.scaled(
                        display_width,
                        display_height,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.result_display.setPixmap(scaled_pixmap)
                else:
                    self.result_display.setPixmap(pixmap)
                self.result_display.setText("")
                self.status_label.setText(f"✅ 图片处理完成")
            else:
                self.result_display.clear()
                self.result_display.setPixmap(QPixmap())
                self.result_display.setText("✅ 视频处理完成\n\n处理后的视频内容")
                self.status_label.setText(f"✅ 视频处理完成")

            # 在右侧文本框中自动填充"脸"
            self.text_input.setText("脸")

            QMessageBox.information(self, "完成", "文件处理完成")

        except Exception as e:
            print(f"处理文件错误: {e}")
            QMessageBox.warning(self, "错误", "处理文件时发生错误")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    login = FancyLogin()
    login.show()

    sys.exit(app.exec_())