import pyautogui
import time

screenWidth, screenHeight = pyautogui.size()
print(screenWidth, screenHeight)

print("开始")
time.sleep(1)
x = 100
y = 150
# 移动鼠标到屏幕坐标(x, y)
pyautogui.moveTo(x, y, duration=1)

time.sleep(1)
x = 200
y = 350
# 可以设置移动鼠标的速度
pyautogui.moveTo(x, y, duration=1)  # 1秒内移动到(x, y)坐标处
pyautogui.doubleClick()

time.sleep(1)
x = 400
y = 550
# 也可以使用相对移动，即基于当前鼠标位置的偏移量
pyautogui.move(y, y, duration=1)  # 在当前位置的基础上，水平移动x_offset，垂直移动y_offset