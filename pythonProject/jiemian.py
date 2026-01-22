import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


class FancyLogin(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Widget")
        self.resize(900, 500)

        # ===== 左右分栏 =====
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        right_panel = QLabel()
        right_panel.setPixmap(QPixmap("tupian/1.jpg").scaled(
            450, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        right_panel.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        # ===== 左侧内容 =====
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(80, 60, 80, 60)

        title = QLabel("LOGIN")
        title.setObjectName("title")
        left_layout.addWidget(title, alignment=Qt.AlignHCenter)

        grid = QGridLayout()
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Username")
        self.user_edit.setObjectName("input")

        self.pwd_edit = QLineEdit()
        self.pwd_edit.setPlaceholderText("Password")
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setObjectName("input")

        grid.addWidget(self.user_edit, 0, 0)
        grid.addWidget(self.pwd_edit, 1, 0)
        left_layout.addLayout(grid)

        self.login_btn = QPushButton("login")
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self.check_login)
        left_layout.addWidget(self.login_btn, alignment=Qt.AlignHCenter)

        hint = QLabel("—— 这里可以放一句宣传语 ——")
        hint.setObjectName("hint")
        left_layout.addWidget(hint, alignment=Qt.AlignHCenter)

        left_layout.addStretch()

        self.setStyleSheet(self.qss())

    def qss(self):
        return """
        QWidget {
            background-color: #fcd0d6;
            font-family: "Segoe UI";
        }
        #title {
            font-size: 28px;
            color: #ff6f91;
            margin-bottom: 20px;
        }
        #input {
            height: 36px;
            border: 2px solid #ffa6c1;
            border-radius: 18px;
            padding-left: 15px;
            margin-top: 10px;
            background: white;
        }
        #input:focus {
            border-color: #ff6f91;
        }
        #login_btn {
            background-color: #ff6f91;
            color: white;
            border-radius: 18px;
            height: 36px;
            width: 120px;
            margin-top: 25px;
        }
        #login_btn:hover {
            background-color: #ff4f7a;
        }
        #hint {
            margin-top: 30px;
            color: #b66;
            font-size: 12px;
        }
        """

    def check_login(self):
        user = self.user_edit.text().strip()
        pwd = self.pwd_edit.text()
        # 这里替换为真实验证逻辑
        if user == "admin" and pwd == "123456":
            print("登录成功")
        else:
            print("用户名或密码错误")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FancyLogin()
    win.show()
    sys.exit(app.exec_())



