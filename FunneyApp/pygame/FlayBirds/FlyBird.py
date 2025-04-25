import pygame
import sys

# 初始化 Pygame
pygame.init()

# 设置屏幕尺寸
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Train Run & Bounce")

# 设置帧率
clock = pygame.time.Clock()
fps = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 128, 255)

# 火车类
class Train:
    def __init__(self, x, y, carriage_count=4):
        self.x = x
        self.y = y
        self.carriage_width = 60
        self.carriage_height = 40
        self.carriage_gap = 8
        self.carriage_count = carriage_count
        self.vel_x = 0  # 初始无自动移动
        self.vel_y = 0
        self.color = (180, 30, 30)
        self.window_color = (220, 220, 255)
        self.wheel_color = (40, 40, 40)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        train_length = self.carriage_count * self.carriage_width + (self.carriage_count-1)*self.carriage_gap
        # 边界检测
        if self.x <= 0:
            self.x = 0
        elif self.x + train_length >= WIDTH:
            self.x = WIDTH - train_length
        if self.y <= 0:
            self.y = 0
        elif self.y + self.carriage_height >= HEIGHT - 30:
            self.y = HEIGHT - 30 - self.carriage_height

    def draw(self, screen):
        for i in range(self.carriage_count):
            cx = self.x + i * (self.carriage_width + self.carriage_gap)
            cy = self.y
            # 画车厢
            pygame.draw.rect(screen, self.color, (cx, cy, self.carriage_width, self.carriage_height), border_radius=8)
            # 画窗户
            pygame.draw.rect(screen, self.window_color, (cx+10, cy+10, self.carriage_width-20, 16), border_radius=4)
            # 画轮子
            pygame.draw.circle(screen, self.wheel_color, (int(cx+12), int(cy+self.carriage_height)), 7)
            pygame.draw.circle(screen, self.wheel_color, (int(cx+self.carriage_width-12), int(cy+self.carriage_height)), 7)
        # 画车头烟囱
        head_x = self.x
        pygame.draw.rect(screen, (80,80,80), (head_x+8, self.y-18, 12, 18), border_radius=4)
        # 画烟雾
        pygame.draw.circle(screen, (200,200,200), (int(head_x+14), int(self.y-24)), 8)
        pygame.draw.circle(screen, (220,220,220), (int(head_x+20), int(self.y-32)), 5)

# 小蚂蚁类
class Ant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 30
        self.vel_x = 0  # 初始无自动移动
        self.vel_y = 0
        self.body_color = (0, 0, 0)
        self.leg_color = (100, 100, 100)
        self.eye_color = (255, 255, 255)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        # 边界检测
        if self.x <= 0:
            self.x = 0
        elif self.x + self.width >= WIDTH:
            self.x = WIDTH - self.width
        if self.y <= 0:
            self.y = 0
        elif self.y + self.height >= HEIGHT - 30:
            self.y = HEIGHT - 30 - self.height

    def draw(self, screen):
        # 画身体
        pygame.draw.ellipse(screen, self.body_color, (self.x, self.y, self.width, self.height))
        # 画眼睛
        pygame.draw.circle(screen, self.eye_color, (int(self.x + 10), int(self.y + 10)), 4)
        pygame.draw.circle(screen, self.eye_color, (int(self.x + self.width - 10), int(self.y + 10)), 4)
        # 画腿
        pygame.draw.line(screen, self.leg_color, (self.x + 5, self.y + self.height), (self.x + 5, self.y + self.height + 10), 2)
        pygame.draw.line(screen, self.leg_color, (self.x + self.width - 5, self.y + self.height), (self.x + self.width - 5, self.y + self.height + 10), 2)

# 小熊猫类
class Panda:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 48
        self.height = 48
        self.vel_x = 0
        self.vel_y = 0
        self.body_color = (255, 255, 255)
        self.ear_color = (30, 30, 30)
        self.eye_color = (30, 30, 30)
        self.nose_color = (80, 40, 40)
        self.cheek_color = (255, 120, 120)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        # 边界检测
        if self.x <= 0:
            self.x = 0
        elif self.x + self.width >= WIDTH:
            self.x = WIDTH - self.width
        if self.y <= 0:
            self.y = 0
        elif self.y + self.height >= HEIGHT - 30:
            self.y = HEIGHT - 30 - self.height

    def draw(self, screen):
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        # 耳朵
        pygame.draw.circle(screen, self.ear_color, (int(cx-16), int(cy-18)), 10)
        pygame.draw.circle(screen, self.ear_color, (int(cx+16), int(cy-18)), 10)
        # 头部
        pygame.draw.ellipse(screen, self.body_color, (self.x, self.y, self.width, self.height))
        # 眼圈
        pygame.draw.ellipse(screen, self.eye_color, (self.x+7, self.y+16, 12, 18))
        pygame.draw.ellipse(screen, self.eye_color, (self.x+29, self.y+16, 12, 18))
        # 眼睛
        pygame.draw.circle(screen, (255,255,255), (int(cx-8), int(cy-2)), 5)
        pygame.draw.circle(screen, (255,255,255), (int(cx+8), int(cy-2)), 5)
        pygame.draw.circle(screen, self.eye_color, (int(cx-8), int(cy-2)), 2)
        pygame.draw.circle(screen, self.eye_color, (int(cx+8), int(cy-2)), 2)
        # 鼻子
        pygame.draw.ellipse(screen, self.nose_color, (int(cx-5), int(cy+8), 10, 6))
        # 腮红
        pygame.draw.ellipse(screen, self.cheek_color, (self.x+6, self.y+32, 8, 5))
        pygame.draw.ellipse(screen, self.cheek_color, (self.x+34, self.y+32, 8, 5))
        # 身体
        pygame.draw.ellipse(screen, self.body_color, (self.x+8, self.y+32, 32, 18))
        # 四肢
        pygame.draw.ellipse(screen, self.ear_color, (self.x+4, self.y+44, 10, 8))
        pygame.draw.ellipse(screen, self.ear_color, (self.x+34, self.y+44, 10, 8))
        pygame.draw.ellipse(screen, self.ear_color, (self.x+4, self.y+28, 10, 8))
        pygame.draw.ellipse(screen, self.ear_color, (self.x+34, self.y+28, 10, 8))

def draw_trees(screen, bg_offset_x):
    # 绘制多棵树，参数可根据需要调整
    tree_positions = [
        (120, 420), (300, 410), (500, 430), (700, 415),
        (900, 420), (1100, 410), (1300, 430), (1500, 415)
    ]
    for x, y in tree_positions:
        tx = x - bg_offset_x
        if -60 < tx < WIDTH+60:  # 只绘制在屏幕范围内的树
            pygame.draw.rect(screen, (110, 70, 30), (tx+18, y+60, 14, 40))
            pygame.draw.circle(screen, (34, 139, 34), (int(tx+25), y+60), 28)
            pygame.draw.circle(screen, (50, 205, 50), (int(tx+15), y+75), 20)
            pygame.draw.circle(screen, (0, 128, 0), (int(tx+35), y+75), 18)

def main():
    ant = Ant(100, HEIGHT - 30 - 40)
    panda = Panda(200, HEIGHT - 30 - 60)
    running = True
    bg_offset_x = 0
    BG_MAX = 1600 - WIDTH  # 背景最大偏移量（假设背景宽1600）

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        # 蚂蚁方向键
        ant.vel_x = (-6 if keys[pygame.K_LEFT] else 6 if keys[pygame.K_RIGHT] else 0)
        ant.vel_y = (-6 if keys[pygame.K_UP] else 6 if keys[pygame.K_DOWN] else 0)
        # 熊猫 WASD
        panda.vel_x = (-6 if keys[pygame.K_a] else 6 if keys[pygame.K_d] else 0)
        panda.vel_y = (-6 if keys[pygame.K_w] else 6 if keys[pygame.K_s] else 0)

        # 背景动态偏移逻辑
        # 优化：两只动物都能推动背景
        move_speed = 6
        # 判断蚂蚁是否推动背景
        if ant.x < 80 and bg_offset_x > 0 and ant.vel_x < 0:
            bg_offset_x -= move_speed
            if bg_offset_x < 0:
                bg_offset_x = 0
            else:
                ant.x = 80
        elif ant.x + ant.width > WIDTH - 80 and bg_offset_x < BG_MAX and ant.vel_x > 0:
            bg_offset_x += move_speed
            if bg_offset_x > BG_MAX:
                bg_offset_x = BG_MAX
            else:
                ant.x = WIDTH - 80 - ant.width
        # 判断熊猫是否推动背景
        if panda.x < 80 and bg_offset_x > 0 and panda.vel_x < 0:
            bg_offset_x -= move_speed
            if bg_offset_x < 0:
                bg_offset_x = 0
            else:
                panda.x = 80
        elif panda.x + panda.width > WIDTH - 80 and bg_offset_x < BG_MAX and panda.vel_x > 0:
            bg_offset_x += move_speed
            if bg_offset_x > BG_MAX:
                bg_offset_x = BG_MAX
            else:
                panda.x = WIDTH - 80 - panda.width

        # 保证动物不会跑出屏幕
        for animal in [ant, panda]:
            if animal.x < 0:
                animal.x = 0
            elif animal.x + animal.width > WIDTH:
                animal.x = WIDTH - animal.width

        # 更新小蚂蚁和小熊猫
        ant.update()
        panda.update()

        # 填充背景色
        screen.fill((200, 200, 200))

        # 画树（根据背景偏移）
        draw_trees(screen, bg_offset_x)

        # 画地面（根据背景偏移）
        pygame.draw.rect(screen, (100, 80, 60), (0, HEIGHT-30, WIDTH, 30))

        # 绘制小蚂蚁
        ant.draw(screen)
        # 绘制小熊猫
        panda.draw(screen)

        # 更新屏幕
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
