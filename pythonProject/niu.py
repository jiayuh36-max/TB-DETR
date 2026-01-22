import sys
import os
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QCheckBox, QFrame, QStackedWidget,
                             QFileDialog, QProgressBar, QGridLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QBrush, QLinearGradient, QPainter, QImage
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QRect, pyqtProperty, QEasingCurve, QThread, pyqtSignal


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度人脸伪造检测系统 - 登录")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("font-family: 'Segoe UI', Arial, sans-serif;")

        # 创建主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧面板
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("""
            #leftPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 #667eea, stop:1 #764ba2);
                border-top-left-radius: 20px;
                border-bottom-left-radius: 20px;
            }
        """)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignCenter)

        # 系统标志和名称
        logo_label = QLabel()
        logo_label.setPixmap(QPixmap("search-icon.png").scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel("深度伪造检测系统")
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("AI-Powered Deepfake Detection System")
        subtitle_label.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 14px; margin-bottom: 20px;")
        subtitle_label.setAlignment(Qt.AlignCenter)

        # 特性列表
        features_widget = QWidget()
        features_layout = QVBoxLayout(features_widget)

        features = [
            ("大脑图标", "先进的深度学习算法"),
            ("图像图标", "支持图像、视频多格式检测"),
            ("图表图标", "实时检测结果可视化"),
            ("盾牌图标", "高精度伪造识别技术"),
            ("时钟图标", "毫秒级快速响应")
        ]

        for icon, text in features:
            feature_layout = QHBoxLayout()
            icon_label = QLabel(icon)  # 实际使用中应该设置图标
            icon_label.setStyleSheet("color: #ffd700;")

            text_label = QLabel(text)
            text_label.setStyleSheet("color: white;")

            feature_layout.addWidget(icon_label)
            feature_layout.addWidget(text_label)
            feature_layout.setAlignment(Qt.AlignLeft)

            features_layout.addLayout(feature_layout)

        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        left_layout.addWidget(logo_label)
        left_layout.addWidget(title_label)
        left_layout.addWidget(subtitle_label)
        left_layout.addWidget(features_widget)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 右侧登录面板
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            background-color: white;
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
        """)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignCenter)

        login_header = QWidget()
        login_header_layout = QVBoxLayout(login_header)

        login_title = QLabel("系统登录")
        login_title.setStyleSheet("color: #333; font-size: 24px; font-weight: bold;")
        login_title.setAlignment(Qt.AlignCenter)

        login_subtitle = QLabel("请输入您的账户信息登录系统")
        login_subtitle.setStyleSheet("color: #666; font-size: 14px; margin-bottom: 20px;")
        login_subtitle.setAlignment(Qt.AlignCenter)

        login_header_layout.addWidget(login_title)
        login_header_layout.addWidget(login_subtitle)

        # 登录表单
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(20)

        # 用户名输入框
        username_label = QLabel("用户名 / 邮箱")
        username_label.setStyleSheet("color: #555; font-weight: 500;")

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名或邮箱")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 12px 12px 12px 40px;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                background: #f8f9fa;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #667eea;
                background: white;
            }
        """)

        # 密码输入框
        password_label = QLabel("密码")
        password_label.setStyleSheet("color: #555; font-weight: 500;")

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 12px 12px 12px 40px;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                background: #f8f9fa;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #667eea;
                background: white;
            }
        """)

        # 记住登录选项
        options_widget = QWidget()
        options_layout = QHBoxLayout(options_widget)
        options_layout.setContentsMargins(0, 0, 0, 0)

        remember_check = QCheckBox("记住登录状态")
        remember_check.setStyleSheet("color: #555;")

        forgot_btn = QPushButton("忘记密码？")
        forgot_btn.setStyleSheet("""
            QPushButton {
                color: #667eea;
                background: transparent;
                border: none;
                font-size: 14px;
                text-align: right;
            }
            QPushButton:hover {
                color: #5a67d8;
            }
        """)
        forgot_btn.setCursor(Qt.PointingHandCursor)

        options_layout.addWidget(remember_check)
        options_layout.addStretch()
        options_layout.addWidget(forgot_btn)

        # 登录按钮
        self.login_btn = QPushButton("立即登录")
        self.login_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #5a67d8, stop:1 #6a3d99);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #4c51bf, stop:1 #553285);
            }
        """)
        self.login_btn.setCursor(Qt.PointingHandCursor)

        # 注册链接
        register_widget = QWidget()
        register_layout = QHBoxLayout(register_widget)
        register_layout.setContentsMargins(0, 0, 0, 0)

        register_text = QLabel("还没有账户？")
        register_text.setStyleSheet("color: #666;")

        register_btn = QPushButton("立即注册")
        register_btn.setStyleSheet("""
            QPushButton {
                color: #667eea;
                background: transparent;
                border: none;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #5a67d8;
            }
        """)
        register_btn.setCursor(Qt.PointingHandCursor)

        register_layout.addStretch()
        register_layout.addWidget(register_text)
        register_layout.addWidget(register_btn)
        register_layout.addStretch()

        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(options_widget)
        form_layout.addWidget(self.login_btn)
        form_layout.addWidget(register_widget)

        right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        right_layout.addWidget(login_header)
        right_layout.addWidget(form_widget)
        right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # 设置窗口阴影效果
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 连接登录按钮到槽
        self.login_btn.clicked.connect(self.login)

    def login(self):
        # 登录按钮点击时的动画效果
        self.login_btn.setText("登录中...")
        self.login_btn.setEnabled(False)

        # 使用定时器模拟登录过程
        QTimer.singleShot(2000, self.show_main_window)

    def show_main_window(self):
        # 创建并显示主界面
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度人脸伪造检测系统")
        self.setGeometry(100, 50, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        # 创建主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        main_layout = QGridLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 创建左侧导航栏
        sidebar = self.create_sidebar()

        # 创建顶部导航栏
        topbar = self.create_topbar()

        # 创建内容区域
        content = self.create_content()

        # 创建右侧分析面板
        analysis_panel = self.create_analysis_panel()

        # 将所有元素添加到主布局
        main_layout.addWidget(sidebar, 0, 0, 3, 1)  # 左侧导航栏
        main_layout.addWidget(topbar, 0, 1, 1, 2)  # 顶部导航栏
        main_layout.addWidget(content, 1, 1, 1, 1)  # 内容区域
        main_layout.addWidget(analysis_panel, 1, 2, 1, 1)  # 右侧分析面板

        # 设置列宽
        main_layout.setColumnStretch(0, 1)  # 侧边栏
        main_layout.setColumnStretch(1, 4)  # 内容区
        main_layout.setColumnStretch(2, 2)  # 分析面板

        # 初始化计时器更新时间
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            #sidebar {
                background-color: #2d3748;
                color: white;
                border-radius: 10px;
                min-width: 220px;
                max-width: 220px;
            }
            QPushButton {
                text-align: left;
                padding: 15px;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #4a5568;
            }
            QPushButton:checked {
                background-color: #4a5568;
                border-left: 5px solid #667eea;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(10)

        # 系统标题
        logo_layout = QVBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("检测系统")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("智能分析工具")
        subtitle.setStyleSheet("color: #a0aec0; font-size: 12px;")
        subtitle.setAlignment(Qt.AlignCenter)

        logo_layout.addWidget(title)
        logo_layout.addWidget(subtitle)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #4a5568; max-height: 1px;")

        # 导航菜单
        detection_label = QLabel("检测功能")
        detection_label.setStyleSheet("color: #a0aec0; font-size: 12px; margin-top: 20px;")

        self.single_btn = QPushButton("单图检测")
        self.single_btn.setCheckable(True)
        self.single_btn.setChecked(True)
        self.single_btn.setIcon(QIcon("image-icon.png"))

        self.batch_btn = QPushButton("批量检测")
        self.batch_btn.setCheckable(True)
        self.batch_btn.setIcon(QIcon("images-icon.png"))

        self.video_btn = QPushButton("视频检测")
        self.video_btn.setCheckable(True)
        self.video_btn.setIcon(QIcon("video-icon.png"))

        settings_label = QLabel("系统设置")
        settings_label.setStyleSheet("color: #a0aec0; font-size: 12px; margin-top: 20px;")

        settings_btn = QPushButton("系统设置")
        settings_btn.setIcon(QIcon("settings-icon.png"))

        history_btn = QPushButton("检测历史")
        history_btn.setIcon(QIcon("history-icon.png"))

        # 添加到布局
        layout.addLayout(logo_layout)
        layout.addWidget(separator)
        layout.addWidget(detection_label)
        layout.addWidget(self.single_btn)
        layout.addWidget(self.batch_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(settings_label)
        layout.addWidget(settings_btn)
        layout.addWidget(history_btn)
        layout.addStretch()

        # 添加退出按钮
        logout_btn = QPushButton("退出登录")
        logout_btn.setIcon(QIcon("logout-icon.png"))
        layout.addWidget(logout_btn)

        # 连接按钮信号到槽函数
        self.single_btn.clicked.connect(lambda: self.switch_mode("single"))
        self.batch_btn.clicked.connect(lambda: self.switch_mode("batch"))
        self.video_btn.clicked.connect(lambda: self.switch_mode("video"))
        logout_btn.clicked.connect(self.close)

        return sidebar

    def create_topbar(self):
        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setStyleSheet("""
            #topbar {
                background-color: white;
                border-radius: 10px;
                padding: 10px;
            }
        """)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("深度人脸伪造检测系统")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")

        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #718096;")
        self.update_time()  # 初始化时间

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.time_label)

        return topbar

    def create_content(self):
        content = QWidget()
        content.setObjectName("content")
        content.setStyleSheet("""
            #content {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
            QPushButton#uploadBtn {
                background-color: #48bb78;
                color: white;
                border-radius: 8px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton#uploadBtn:hover {
                background-color: #38a169;
            }
            QLabel#resultScore {
                font-size: 36px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(content)
        layout.setSpacing(20)

        # 上传区域
        upload_widget = QWidget()
        upload_widget.setObjectName("uploadWidget")
        upload_widget.setStyleSheet("""
            #uploadWidget {
                background-color: #f7fafc;
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                padding: 40px;
                min-height: 250px;
            }
            #uploadWidget:hover {
                border-color: #667eea;
                background-color: #edf2f7;
            }
        """)

        upload_layout = QVBoxLayout(upload_widget)
        upload_layout.setAlignment(Qt.AlignCenter)

        upload_icon = QLabel("📁")  # 简单使用emoji作为图标
        upload_icon.setStyleSheet("font-size: 32px;")
        upload_icon.setAlignment(Qt.AlignCenter)

        upload_text = QLabel("点击或拖拽上传图片")
        upload_text.setStyleSheet("color: #718096; font-size: 16px;")
        upload_text.setAlignment(Qt.AlignCenter)

        upload_format = QLabel("支持格式：JPG, PNG, BMP")
        upload_format.setStyleSheet("color: #718096; font-size: 12px;")
        upload_format.setAlignment(Qt.AlignCenter)

        upload_btn = QPushButton("选择文件")
        upload_btn.setObjectName("uploadBtn")
        upload_btn.setFixedWidth(120)
        upload_btn.clicked.connect(self.open_file_dialog)

        upload_layout.addWidget(upload_icon)
        upload_layout.addWidget(upload_text)
        upload_layout.addWidget(upload_format)
        upload_layout.addSpacing(20)
        upload_layout.addWidget(upload_btn, 0, Qt.AlignCenter)

        # 预览图像
        self.preview_image = QLabel()
        self.preview_image.setObjectName("previewImage")
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumHeight(200)
        self.preview_image.setStyleSheet("""
            #previewImage {
                background-color: #f7fafc;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #e2e8f0;
            }
        """)
        self.preview_image.hide()  # 初始时隐藏

        # 检测结果区域
        results_widget = QWidget()
        results_widget.setObjectName("resultsWidget")
        results_widget.setStyleSheet("""
            #resultsWidget {
                background-color: #f7fafc;
                border-radius: 8px;
                padding: 20px;
                border:1px solid #e2e8f0;
            }
        """)
        results_layout = QVBoxLayout(results_widget)

        results_title = QLabel("检测结果")
        results_title.setStyleSheet("font-weight: bold; font-size: 16px;")

        results_detail = QWidget()
        results_detail_layout = QHBoxLayout(results_detail)
        results_detail_layout.setContentsMargins(0, 10, 0, 10)

        self.result_score = QLabel("0%")
        self.result_score.setObjectName("resultScore")
        self.result_score.setStyleSheet("color: #48bb78;")  # 默认绿色

        result_desc_widget = QWidget()
        result_desc_layout = QVBoxLayout(result_desc_widget)
        result_desc_layout.setContentsMargins(0, 0, 0, 0)

        self.result_description = QLabel("等待检测结果...")
        self.result_description.setStyleSheet("color: #4a5568;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #e2e8f0;
                border-radius: 5px;
                height: 10px;
                margin-top: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #48bb78, stop:1 #38a169);
                border-radius: 5px;
            }
        """)

        result_desc_layout.addWidget(self.result_description)
        result_desc_layout.addWidget(self.progress_bar)

        results_detail_layout.addWidget(self.result_score)
        results_detail_layout.addWidget(result_desc_widget)

        # 警告框
        self.alert_box = QWidget()
        self.alert_box.setObjectName("alertBox")
        self.alert_box.setStyleSheet("""
            #alertBox {
                background-color: #fed7d7;
                border: 1px solid #feb2b2;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        alert_layout = QHBoxLayout(self.alert_box)

        alert_icon = QLabel("⚠️")  # 使用emoji作为图标
        alert_text = QLabel("检测到高风险伪造！")
        alert_text.setStyleSheet("color: #c53030; font-weight: bold;")

        alert_layout.addWidget(alert_icon)
        alert_layout.addWidget(alert_text)
        alert_layout.addStretch()

        self.alert_box.hide()  # 初始时隐藏

        # 添加所有部件到主布局
        results_layout.addWidget(results_title)
        results_layout.addWidget(results_detail)
        results_layout.addWidget(self.alert_box)

        layout.addWidget(upload_widget)
        layout.addWidget(self.preview_image)
        layout.addWidget(results_widget)
        layout.addStretch()

        return content

    def create_analysis_panel(self):
        panel = QWidget()
        panel.setObjectName("analysisPanel")
        panel.setStyleSheet("""
            #analysisPanel {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
            }
            QLabel.metric-label {
                color: #4a5568;
                font-weight: bold;
            }
            QLabel.metric-value {
                color: #2d3748;
                font-weight: bold;
            }
            QWidget.metric {
                background-color: #f7fafc;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                padding: 10px;
            }
        """)

        layout = QVBoxLayout(panel)

        title = QLabel("详细分析")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")

        # 创建指标部件
        metrics = [
            ("面部一致性", "faceConsistency"),
            ("光照分析", "lightingAnalysis"),
            ("几何变形", "geometryDistortion"),
            ("伪造置信度", "fakeConfidence")
        ]

        self.metrics = {}

        for label_text, metric_id in metrics:
            metric_widget = QWidget()
            metric_widget.setObjectName(f"metric_{metric_id}")
            metric_widget.setProperty("class", "metric")

            metric_layout = QHBoxLayout(metric_widget)
            metric_layout.setContentsMargins(10, 10, 10, 10)

            metric_label = QLabel(f"{label_text}:")
            metric_label.setProperty("class", "metric-label")

            metric_value = QLabel("N/A")
            metric_value.setObjectName(metric_id)
            metric_value.setProperty("class", "metric-value")

            metric_layout.addWidget(metric_label)
            metric_layout.addStretch()
            metric_layout.addWidget(metric_value)

            layout.addWidget(metric_widget)
            self.metrics[metric_id] = metric_value

        layout.addStretch()

        # 添加标题到布局
        layout.insertWidget(0, title)

        return panel

    def update_time(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"当前时间: {current_time}")

    def switch_mode(self, mode):
        # 取消所有按钮的选中状态
        self.single_btn.setChecked(False)
        self.batch_btn.setChecked(False)
        self.video_btn.setChecked(False)

        # 设置选中的按钮
        if mode == "single":
            self.single_btn.setChecked(True)
        elif mode == "batch":
            self.batch_btn.setChecked(True)
        elif mode == "video":
            self.video_btn.setChecked(True)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            # 加载并显示图像
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 缩放图片以适应预览区域，保持宽高比
                pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_image.setPixmap(pixmap)
                self.preview_image.show()

                # 模拟检测过程
                self.perform_detection()

    def perform_detection(self):
        # 重置分析面板
        for metric_id in self.metrics:
            self.metrics[metric_id].setText("分析中...")

        # 模拟处理延迟
        QTimer.singleShot(1000, self.update_detection_results)

    def update_detection_results(self):
        # 生成随机检测结果
        import random
        score = random.randint(0, 100)

        # 更新进度条
        self.progress_bar.setValue(score)

        # 更新结果分数和描述
        self.result_score.setText(f"{score}%")

        # 根据分数设置不同的颜色和描述
        if score >= 80:
            self.result_score.setStyleSheet("color: #c53030;")  # 红色，伪造
            self.result_description.setText("图像被检测为可能伪造")
            self.alert_box.show()
        elif score >= 50:
            self.result_score.setStyleSheet("color: #ed8936;")  # 橙色，可疑
            self.result_description.setText("图像检测结果存在疑点")
            self.alert_box.hide()
        else:
            self.result_score.setStyleSheet("color: #48bb78;")  # 绿色，真实
            self.result_description.setText("图像被检测为真实")
            self.alert_box.hide()

        # 更新分析面板的指标
        face_consistency = random.randint(0, 100)
        lighting = random.randint(0, 100)
        geometry = random.randint(0, 100)

        self.metrics["faceConsistency"].setText(f"{face_consistency}%")
        self.metrics["lightingAnalysis"].setText(f"{lighting}%")
        self.metrics["geometryDistortion"].setText(f"{geometry}%")
        self.metrics["fakeConfidence"].setText(f"{score}%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
