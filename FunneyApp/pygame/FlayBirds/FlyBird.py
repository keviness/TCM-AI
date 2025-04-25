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

def main():
    train = Train(100, HEIGHT - 30 - 40)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    train.vel_x = -6
                elif event.key == pygame.K_RIGHT:
                    train.vel_x = 6
                elif event.key == pygame.K_UP:
                    train.vel_y = -6
                elif event.key == pygame.K_DOWN:
                    train.vel_y = 6
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    train.vel_x = 0
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    train.vel_y = 0

        # 更新火车
        train.update()

        # 填充背景色
        screen.fill((200, 200, 200))

        # 画地面
        pygame.draw.rect(screen, (100, 80, 60), (0, HEIGHT-30, WIDTH, 30))

        # 绘制火车
        train.draw(screen)

        # 更新屏幕
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
